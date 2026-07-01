from __future__ import annotations

import json
import re

from agentenv_agentmemory.environment import AgentMemoryEnv


PRODUCT_RE = re.compile(r"^- (?P<product_id>\w+): (?P<title>.+?) \((?P<attrs>.+)\)$")


def parse_products(observation: str) -> dict[str, dict[str, str]]:
    products: dict[str, dict[str, str]] = {}
    for line in observation.splitlines():
        match = PRODUCT_RE.match(line.strip())
        if not match:
            continue
        attrs = {}
        for part in match.group("attrs").split(", "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            attrs[key] = value
        products[match.group("product_id")] = attrs
    return products


def action(op: str, **payload) -> str:
    return f"{op} {json.dumps(payload, ensure_ascii=False)}"


class LatestObservationMemoryPolicy:
    def __init__(self) -> None:
        self.memory: dict[str, str] = {}
        self.added_keys: set[str] = set()
        self.pending_buy: str | None = None

    def act(self, observation: str) -> str:
        products = parse_products(observation)
        if self.pending_buy:
            product_id = self.pending_buy
            self.pending_buy = None
            return action("BUY", product_id=product_id)

        if "Active short-term memory/context: <empty>" not in observation:
            return self.buy_from_retrieved_context(observation, products)
        if "larger-screen TV" in observation:
            product_id = self.pick_product_with_attr(products, "tv_size_in", "75")
            attrs = products[product_id]
            value = f"Purchased TV: {attrs['tv_size_in']} inches, {attrs['tv_weight_kg']}kg, VESA {attrs['vesa']}."
            return self.add_then_buy("tv_profile", value, product_id)
        if "14-inch laptop" in observation:
            product_id = self.pick_product_with_attr(products, "laptop_size_in", "14")
            attrs = products[product_id]
            value = f"Purchased laptop: {attrs['laptop_size_in']} inches with {attrs['laptop_ports']} port."
            return self.add_then_buy("laptop_profile", value, product_id)
        if "usb-c support" in observation:
            product_id = self.pick_product_with_attr(products, "monitor_ports", "usb-c")
            attrs = products[product_id]
            value = (
                f"Purchased monitor: {attrs['monitor_size_in']} inches, {attrs['monitor_weight_kg']}kg, "
                f"VESA {attrs['vesa']}, ports {attrs['monitor_ports']}."
            )
            return self.add_then_buy("monitor_profile", value, product_id)
        if "wall mount" in observation or "media console" in observation:
            return action("RETRIEVE", query=self.memory["tv_profile"], top_k=1)
        if "sleeve" in observation or "dock" in observation:
            return action("RETRIEVE", query=self.memory["laptop_profile"], top_k=1)
        if "monitor arm" in observation or "display cable" in observation:
            return action("RETRIEVE", query=self.memory["monitor_profile"], top_k=1)
        raise AssertionError(f"Unhandled observation:\\n{observation}")

    def add_then_buy(self, key: str, value: str, product_id: str) -> str:
        self.memory[key] = value
        if key not in self.added_keys:
            self.added_keys.add(key)
            self.pending_buy = product_id
            return action("ADD", key=key, value=value)
        return action("BUY", product_id=product_id)

    def buy_from_retrieved_context(self, observation: str, products: dict[str, dict[str, str]]) -> str:
        if "Purchased TV: 75 inches" in observation:
            if any("compatible_tv_max" in attrs for attrs in products.values()):
                return action("BUY", product_id=self.pick_product_with_attr(products, "compatible_tv_max", "85"))
            return action("BUY", product_id=self.pick_product_with_attr(products, "compatible_tv_min", "70"))
        if "Purchased laptop: 14 inches" in observation:
            if any("compatible_laptop_max" in attrs for attrs in products.values()):
                return action("BUY", product_id=self.pick_product_with_attr(products, "compatible_laptop_min", "14"))
            return action("BUY", product_id=self.pick_product_with_attr(products, "required_port", "usb-c"))
        if "Purchased monitor: 27 inches" in observation:
            if any("compatible_monitor_max" in attrs for attrs in products.values()):
                return action("BUY", product_id=self.pick_product_with_attr(products, "compatible_monitor_max", "32"))
            return action("BUY", product_id=self.pick_product_with_attr(products, "required_port", "usb-c"))
        raise AssertionError(f"Retrieved context is insufficient:\\n{observation}")

    @staticmethod
    def pick_product_with_attr(products: dict[str, dict[str, str]], attr: str, expected: str) -> str:
        for product_id, attrs in products.items():
            if attrs.get(attr) == expected or expected in attrs.get(attr, ""):
                return product_id
        raise AssertionError(f"No product has {attr}={expected}: {products}")


def run_policy(data_idx: int) -> dict:
    env = AgentMemoryEnv()
    observation, info = env.reset(data_idx=data_idx)
    policy = LatestObservationMemoryPolicy()
    reward_sum = 0.0
    done = False
    action_ops: list[str] = []
    memory_ops: list[str] = []
    for _ in range(12):
        chosen_action = policy.act(observation)
        action_ops.append(chosen_action.split(maxsplit=1)[0])
        observation, reward, done, _, info = env.step(chosen_action)
        memory_ops.extend(item["op"] for item in info["memory_ops"])
        reward_sum += reward
        if done:
            break
    assert done, f"latest-observation policy did not finish data_idx={data_idx}: {info}"
    assert info["episode_success"], info
    assert info["progress_score"] == 1.0, info
    assert "ADD" in action_ops and "RETRIEVE" in action_ops and "BUY" in action_ops, action_ops
    assert "ADD" in memory_ops and "RETRIEVE" in memory_ops, memory_ops
    assert reward_sum > 0, reward_sum
    return info


def assert_no_memory_policy_fails() -> None:
    env = AgentMemoryEnv()
    observation, _ = env.reset(data_idx=0)
    products = parse_products(observation)
    tv = LatestObservationMemoryPolicy.pick_product_with_attr(products, "tv_size_in", "75")
    observation, _, _, _, _ = env.step(action("BUY", product_id=tv))
    products = parse_products(observation)
    wrong_mount = LatestObservationMemoryPolicy.pick_product_with_attr(products, "compatible_tv_max", "60")
    _, reward, done, _, info = env.step(action("BUY", product_id=wrong_mount))
    assert reward < 0 and not done and info["compatibility_violations"], info


def main() -> None:
    infos = [run_policy(idx) for idx in range(3)]
    assert_no_memory_policy_fails()
    print("AGENTMEMORY_LATEST_OBSERVATION_POLICY_SMOKE_OK", " ".join(info["task_id"] for info in infos))


if __name__ == "__main__":
    main()
