MACHINE HighwayEnvironment2

DEFINITIONS
    SHIELD_INTERVENTION == {
      IDLE |-> IF ENABLED("IDLE") = TRUE
               THEN
                 IDLE
               ELSIF ENABLED("LANE_LEFT") = TRUE
               THEN
                 LANE_LEFT
               ELSIF ENABLED("LANE_RIGHT") = TRUE
               THEN
                 LANE_RIGHT
               ELSE
                 SLOWER
               END,
      FASTER |-> IF ENABLED("FASTER") = TRUE
                 THEN
                   FASTER
                 ELSIF ENABLED("IDLE") = TRUE
                 THEN
                   IDLE
                 ELSIF ENABLED("LANE_LEFT") = TRUE
                 THEN
                   LANE_LEFT
                 ELSIF ENABLED("LANE_RIGHT") = TRUE
                 THEN
                   LANE_RIGHT
                 ELSE
                   SLOWER
                 END,
      SLOWER |-> IF #v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 15.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5) THEN
		   IF ENABLED("LANE_LEFT") = TRUE THEN LANE_LEFT
		   ELSIF ENABLED("LANE_RIGHT") = TRUE THEN LANE_RIGHT
		   ELSE SLOWER
		 END
		 ELSE
		   SLOWER
		 END,
      LANE_LEFT |-> IF ENABLED("LANE_LEFT") = TRUE
                    THEN
                      LANE_LEFT
                    ELSIF ENABLED("IDLE") = TRUE
                    THEN
                      IDLE
                    ELSIF ENABLED("LANE_RIGHT") = TRUE
                    THEN
                      LANE_RIGHT
                    ELSE
                      SLOWER
                    END,
      LANE_RIGHT |-> IF ENABLED("LANE_RIGHT") = TRUE
                     THEN
                       LANE_RIGHT
                     ELSIF ENABLED("IDLE") = TRUE
                     THEN  
                       IDLE
                     ELSIF ENABLED("LANE_LEFT") = TRUE
                     THEN
                       LANE_LEFT
                     ELSE
                       SLOWER
                     END
    }

SETS
    Vehicles;
    Actions = {IDLE, SLOWER, FASTER, LANE_LEFT, LANE_RIGHT}

CONSTANTS
    EgoVehicle
PROPERTIES
    card(Vehicles) > 1 &
    card(Vehicles) = 5 &
    EgoVehicle : Vehicles

VARIABLES
    Crash,
    VehiclesX,
    VehiclesY,
    VehiclesVx,
    VehiclesVy,
    VehiclesAx,
    VehiclesAy,
    Reward

INVARIANT
    Crash : BOOL &
    VehiclesX : Vehicles --> REAL &
    VehiclesY : Vehicles --> REAL &
    VehiclesVx : Vehicles --> REAL &
    VehiclesVy : Vehicles --> REAL &
    VehiclesAx : Vehicles --> REAL &
    VehiclesAy : Vehicles --> REAL &
    Reward : REAL

INITIALISATION
  Crash := FALSE ||
  VehiclesX : (VehiclesX : Vehicles --> REAL) ||
  VehiclesY : (VehiclesY : Vehicles --> REAL) ||
  VehiclesVx : (VehiclesVx : Vehicles --> REAL) ||
  VehiclesVy : (VehiclesVy : Vehicles --> REAL) ||
  VehiclesAx : (VehiclesAx : Vehicles --> REAL) ||
  VehiclesAy : (VehiclesAy : Vehicles --> REAL) ||
  Reward : (Reward : REAL)

OPERATIONS
  IDLE =
  SELECT
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 35.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5 & VehiclesVx(v) < 0.0))
  THEN
      Crash :: BOOL ||
      VehiclesX : (VehiclesX : Vehicles --> REAL) ||
      VehiclesY : (VehiclesY : Vehicles --> REAL) ||
      VehiclesVx : (VehiclesVx : Vehicles --> REAL) ||
      VehiclesVy : (VehiclesVy : Vehicles --> REAL) ||
      VehiclesAx : (VehiclesAx : Vehicles --> REAL) ||
      VehiclesAy : (VehiclesAy : Vehicles --> REAL) ||
      Reward : (Reward : REAL)
  END;

  LANE_LEFT =
  SELECT
    VehiclesY(EgoVehicle) > 1.0 &
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) > 0.0)) &
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) <= 15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0)) &
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) < 0.0))
  THEN
      Crash :: BOOL ||
      VehiclesX : (VehiclesX : Vehicles --> REAL) ||
      VehiclesY : (VehiclesY : Vehicles --> REAL) ||
      VehiclesVx : (VehiclesVx : Vehicles --> REAL) ||
      VehiclesVy : (VehiclesVy : Vehicles --> REAL) ||
      VehiclesAx : (VehiclesAx : Vehicles --> REAL) ||
      VehiclesAy : (VehiclesAy : Vehicles --> REAL) ||
      Reward : (Reward : REAL)
  END;

  LANE_RIGHT =
  SELECT
    VehiclesY(EgoVehicle) < 7.0 &
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) > 0.0)) &
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) < 15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0)) &
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) < 0.0))
  THEN
      Crash :: BOOL ||
      VehiclesX : (VehiclesX : Vehicles --> REAL) ||
      VehiclesY : (VehiclesY : Vehicles --> REAL) ||
      VehiclesVx : (VehiclesVx : Vehicles --> REAL) ||
      VehiclesVy : (VehiclesVy : Vehicles --> REAL) ||
      VehiclesAx : (VehiclesAx : Vehicles --> REAL) ||
      VehiclesAy : (VehiclesAy : Vehicles --> REAL) ||
      Reward : (Reward : REAL)
  END;

  FASTER =
  SELECT
    not(#v. (v : Vehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 45.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5))
  THEN
      Crash :: BOOL ||
      VehiclesX : (VehiclesX : Vehicles --> REAL) ||
      VehiclesY : (VehiclesY : Vehicles --> REAL) ||
      VehiclesVx : (VehiclesVx : Vehicles --> REAL) ||
      VehiclesVy : (VehiclesVy : Vehicles --> REAL) ||
      VehiclesAx : (VehiclesAx : Vehicles --> REAL) ||
      VehiclesAy : (VehiclesAy : Vehicles --> REAL) ||
      Reward : (Reward : REAL)
  END;

  SLOWER =
  BEGIN
      Crash :: BOOL ||
      VehiclesX : (VehiclesX : Vehicles --> REAL) ||
      VehiclesY : (VehiclesY : Vehicles --> REAL) ||
      VehiclesVx : (VehiclesVx : Vehicles --> REAL) ||
      VehiclesVy : (VehiclesVy : Vehicles --> REAL) ||
      VehiclesAx : (VehiclesAx : Vehicles --> REAL) ||
      VehiclesAy : (VehiclesAy : Vehicles --> REAL) ||
      Reward : (Reward : REAL)
  END

END