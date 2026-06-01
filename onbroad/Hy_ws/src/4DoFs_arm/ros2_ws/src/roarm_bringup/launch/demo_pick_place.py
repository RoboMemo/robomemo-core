"""
demo_pick_place.launch.py

Demonstrates a continuous pick-and-place cycle:
1. Trigger detection
2. Send the first detected object to /pick_place/execute action
3. Place at a fixed drop zone
4. Repeat

Expects the full bringup.launch.py to already be running.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseArray
from std_srvs.srv import Trigger
from roarm_pick_place.action import PickPlace

import time


class DemoPickPlace(Node):
    def __init__(self):
        super().__init__("demo_pick_place")

        self.declare_parameter("place_x", 0.00)
        self.declare_parameter("place_y", 0.22)
        self.declare_parameter("place_z", 0.05)
        self.declare_parameter("num_cycles", 5)

        self.place_x = self.get_parameter("place_x").value
        self.place_y = self.get_parameter("place_y").value
        self.place_z = self.get_parameter("place_z").value
        self.num_cycles = self.get_parameter("num_cycles").value

        self._action_client = ActionClient(self, PickPlace, "/pick_place/execute")
        self._trigger_client = self.create_client(Trigger, "/perception/trigger_detection")
        self._latest_objects = None

        self._det_sub = self.create_subscription(
            PoseArray,
            "/perception/detected_objects",
            self._det_cb, 10)

        self.get_logger().info("DemoPickPlace waiting for action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action server ready. Starting demo.")

        # Start demo after a short delay
        self._timer = self.create_timer(2.0, self._run_demo)

    def _det_cb(self, msg: PoseArray):
        self._latest_objects = msg

    def _trigger_detection(self):
        """Call trigger and wait up to 5s for detection result."""
        if self._trigger_client.wait_for_service(timeout_sec=3.0):
            future = self._trigger_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        # Wait for poses
        deadline = time.time() + 5.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_objects and len(self._latest_objects.poses) > 0:
                return self._latest_objects.poses[0]
        return None

    def _run_demo(self):
        self._timer.cancel()  # run once

        for cycle in range(self.num_cycles):
            self.get_logger().info(f"=== Cycle {cycle + 1}/{self.num_cycles} ===")

            pose = self._trigger_detection()
            if pose is None:
                self.get_logger().warn("No object detected, skipping cycle.")
                time.sleep(2.0)
                continue

            self.get_logger().info(
                f"Target: ({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})")

            goal = PickPlace.Goal()
            goal.target_pose  = pose
            goal.place_x      = self.place_x
            goal.place_y      = self.place_y
            goal.place_z      = self.place_z
            goal.gripper_open_rad  = 1.5
            goal.gripper_close_rad = 0.2

            future = self._action_client.send_goal_async(
                goal,
                feedback_callback=lambda fb:
                    self.get_logger().info(
                        f"  [{fb.feedback.current_state}] "
                        f"{fb.feedback.progress_percent:.0f}%"))

            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().error("Goal rejected.")
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
            result = result_future.result().result

            if result.success:
                self.get_logger().info(f"Cycle {cycle + 1} SUCCESS: {result.message}")
            else:
                self.get_logger().error(f"Cycle {cycle + 1} FAILED: {result.message}")

            time.sleep(1.0)

        self.get_logger().info("Demo complete.")


def main():
    rclpy.init()
    node = DemoPickPlace()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
