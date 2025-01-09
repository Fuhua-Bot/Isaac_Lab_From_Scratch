import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg, ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg


BIP_WL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/Projects/BipedalWheeledRobot/BipedalWheeledRobotV1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={"LeftPassiveDrive": 0.0, "LeftMotorDrive": 0.0, "RightMotorDrive": 0.0, "RightPassiveDrive": 0.0},
        joint_vel={"LeftWheel": 0.0, "RightWheel": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "wheels": DCMotorCfg(
            joint_names_expr=[
                "LeftWheel",
                "RightWheel",
            ],
            saturation_effort=3.5,
            effort_limit=3.5,
            velocity_limit=40.0,
            stiffness=100.0,
            damping=4,
            friction=0.0,
        ),
        "legs": DCMotorCfg(
            joint_names_expr=[
                "LeftPassiveDrive",
                "LeftMotorDrive",
                "RightMotorDrive",
                "RightPassiveDrive",
            ],
            saturation_effort=33.5,
            effort_limit=33.5,
            velocity_limit=30.0,
            stiffness=50.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
