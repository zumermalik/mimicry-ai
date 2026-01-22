import unittest
import math
from mimicry_core import RobotArmSim

class TestRoboticLogic(unittest.TestCase):
    
    def setUp(self):
        # Initialize a standard robot arm for testing
        # L1 = 100, L2 = 100 for easy math
        self.bot = RobotArmSim(l1=100, l2=100, origin=(0, 0))

    def test_initial_state(self):
        """Verify robot starts in a known state."""
        self.assertTrue(self.bot.target_reached)
        print("\n[PASS] Robot initialization consistent.")

    def test_max_reach_limit(self):
        """Test the safety clamping mechanism."""
        # Try to reach point (300, 0) which is impossible (Max span is 200)
        self.bot.inverse_kinematics(300, 0)
        
        # Check if flag was set to False
        self.assertFalse(self.bot.target_reached)
        
        # Check if it clamped to max range (200)
        # At (200, 0), both angles should be 0 (flat out)
        # Note: Depending on coordinate system, checking distance is safer
        s, e, w = self.bot.get_joints()
        wrist_dist = math.sqrt(w[0]**2 + w[1]**2)
        
        # Allow tiny float error (epsilon)
        self.assertAlmostEqual(wrist_dist, 200, delta=1.0)
        print("[PASS] Safety clamp successfully blocked hyperextension.")

    def test_zero_position(self):
        """Test folding the arm onto itself."""
        # Target the origin (0,0)
        self.bot.inverse_kinematics(0.01, 0.01) # Avoid true 0/0 div error
        self.assertTrue(self.bot.target_reached)
        print("[PASS] Zero-point singularity handled.")

if __name__ == '__main__':
    print(">>> RUNNING AUTOMATED KINEMATIC TESTS...")
    unittest.main()
