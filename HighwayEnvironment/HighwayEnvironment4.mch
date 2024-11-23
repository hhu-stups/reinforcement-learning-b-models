/*
Developed by Fabian Vu
*/
MACHINE HighwayEnvironment2

SETS
    Vehicles

CONSTANTS
    EgoVehicle
PROPERTIES
    card(Vehicles) > 1 &
    card(Vehicles) = 5 &
    EgoVehicle : Vehicles

VARIABLES
    Crash,
    PresentVehicles,
    VehiclesX,
    VehiclesY,
    VehiclesVx,
    VehiclesVy,
    VehiclesAx,
    VehiclesAy,
    Reward

INVARIANT
    Crash : BOOL &
    PresentVehicles : POW(Vehicles) &
    EgoVehicle : PresentVehicles &
    VehiclesX : PresentVehicles --> REAL &
    VehiclesY : PresentVehicles --> REAL &
    VehiclesVx : PresentVehicles --> REAL &
    VehiclesVy : PresentVehicles --> REAL &
    VehiclesAx : PresentVehicles --> REAL &
    VehiclesAy : PresentVehicles --> REAL &
    Reward : REAL

INITIALISATION
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles);
    VehiclesX :: PresentVehicles --> REAL ||
    VehiclesY :: PresentVehicles --> REAL ||
    VehiclesVx :: PresentVehicles --> REAL ||
    VehiclesVy :: PresentVehicles --> REAL ||
    VehiclesAx :: PresentVehicles --> REAL ||
    VehiclesAy :: PresentVehicles --> REAL ||
    Reward :: REAL

OPERATIONS
  IDLE(ActiveVehicles) =
  SELECT
    ActiveVehicles : POW(Vehicles) & EgoVehicle : ActiveVehicles &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 35.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5 & VehiclesVx(v) < 0.0))
  THEN
    Crash :: BOOL ||
    PresentVehicles := ActiveVehicles ||
    VehiclesX :: ActiveVehicles --> REAL ||
    VehiclesY :: ActiveVehicles --> REAL ||
    VehiclesVx :: ActiveVehicles --> REAL ||
    VehiclesVy :: ActiveVehicles --> REAL ||
    VehiclesAx :: ActiveVehicles --> REAL ||
    VehiclesAy :: ActiveVehicles --> REAL ||
    Reward :: REAL
  END;

  LANE_LEFT(ActiveVehicles) =
  SELECT
    ActiveVehicles : POW(Vehicles) & EgoVehicle : ActiveVehicles &
    VehiclesY(EgoVehicle) > 1.0 &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) > 0.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) <= 15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) < 0.0))
  THEN
    Crash :: BOOL ||
    PresentVehicles := ActiveVehicles ||
    VehiclesX :: ActiveVehicles --> REAL ||
    VehiclesY :(VehiclesY : ActiveVehicles --> REAL & (Crash = TRUE or (VehiclesY(EgoVehicle) <= VehiclesY$0(EgoVehicle) + 0.05)))  ||
    VehiclesVx :: ActiveVehicles --> REAL ||
    VehiclesVy :: ActiveVehicles --> REAL ||
    VehiclesAx :: ActiveVehicles --> REAL ||
    VehiclesAy :: ActiveVehicles --> REAL ||
    Reward :: REAL
  END;

  LANE_RIGHT(ActiveVehicles) =
  SELECT
    ActiveVehicles : POW(Vehicles) & EgoVehicle : ActiveVehicles &
    VehiclesY(EgoVehicle) < 7.0 &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) > 0.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) < 15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) < 0.0))
  THEN
    Crash :: BOOL ||
    PresentVehicles := ActiveVehicles ||
    VehiclesX :: ActiveVehicles --> REAL ||
    VehiclesY :(VehiclesY : ActiveVehicles --> REAL & (Crash = TRUE or (VehiclesY(EgoVehicle) >= VehiclesY$0(EgoVehicle) - 0.05))) ||
    VehiclesVx :: ActiveVehicles --> REAL ||
    VehiclesVy :: ActiveVehicles --> REAL ||
    VehiclesAx :: ActiveVehicles --> REAL ||
    VehiclesAy :: ActiveVehicles --> REAL ||
    Reward :: REAL
  END;

  FASTER(ActiveVehicles) =
  SELECT
    ActiveVehicles : POW(Vehicles) & EgoVehicle : ActiveVehicles &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 45.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5))
  THEN
    Crash :: BOOL ||
    PresentVehicles := ActiveVehicles ||
    VehiclesX :: ActiveVehicles --> REAL ||
    VehiclesY :: ActiveVehicles --> REAL ||
    VehiclesVx : (VehiclesVx : ActiveVehicles --> REAL & (Crash = TRUE or VehiclesVx(EgoVehicle) >= VehiclesVx$0(EgoVehicle) - 0.05)) ||
    VehiclesVy :: ActiveVehicles --> REAL ||
    VehiclesAx : (VehiclesAx : ActiveVehicles --> REAL & (Crash = TRUE or VehiclesAx(EgoVehicle) >= -0.05)) ||
    VehiclesAy :: ActiveVehicles --> REAL ||
    Reward :: REAL
  END;

  SLOWER(ActiveVehicles) =
  SELECT
    ActiveVehicles : POW(Vehicles) & EgoVehicle : ActiveVehicles &
    (#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 15.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5 & VehiclesVx(v) < 0.0)
    => 
    (not(
    VehiclesY(EgoVehicle) > 1.0 &
      not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) > 0.0)) &
      not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) <= 15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0)) &
      not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) < 0.0))
    ) &
    not(
      VehiclesY(EgoVehicle) < 7.0 &
      not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) > 0.0)) &
      not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) < 15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0)) &
      not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) < 0.0))
    )))
  THEN
    Crash :: BOOL ||
    PresentVehicles := ActiveVehicles ||
    VehiclesX :: ActiveVehicles --> REAL ||
    VehiclesY :: ActiveVehicles --> REAL ||
    VehiclesVx : (VehiclesVx : ActiveVehicles --> REAL & (Crash = TRUE or VehiclesVx(EgoVehicle) <= VehiclesVx$0(EgoVehicle) + 0.05)) ||
    VehiclesVy :: ActiveVehicles --> REAL ||
    VehiclesAx : (VehiclesAx : ActiveVehicles --> REAL & (Crash = TRUE or VehiclesAx(EgoVehicle) <= 0.05)) ||
    VehiclesAy :: ActiveVehicles --> REAL ||
    Reward :: REAL
  END

END