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
    VehiclesX : Vehicles --> REAL &
    VehiclesY : Vehicles --> REAL &
    VehiclesVx : Vehicles --> REAL &
    VehiclesVy : Vehicles --> REAL &
    VehiclesAx : Vehicles --> REAL &
    VehiclesAy : Vehicles --> REAL &
    Reward : REAL

INITIALISATION
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL ||
    VehiclesY :: Vehicles --> REAL ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL

OPERATIONS
  IDLE =
  SELECT
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 35.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5 & VehiclesVx(v) < 0.0))
  THEN
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL ||
    VehiclesY :: Vehicles --> REAL ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL
  END;

  LANE_LEFT =
  SELECT
    EgoVehicle : dom(VehiclesY) &
    VehiclesY(EgoVehicle) > 1.0 &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) > 0.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) <= 15.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) > -6.0 & VehiclesY(v) < -2.0 & VehiclesVx(v) < 0.0))
  THEN
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL ||
    VehiclesY :(VehiclesY : Vehicles --> REAL & (Crash = TRUE or (VehiclesY(EgoVehicle) <= VehiclesY$0(EgoVehicle) + 0.05)))  ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL
  END;

  LANE_RIGHT =
  SELECT
    EgoVehicle : dom(VehiclesY) &
    VehiclesY(EgoVehicle) < 7.0 &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -25.0 & VehiclesX(v) <= -15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) > 0.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > -15.0 & VehiclesX(v) < 15.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 15.0 & VehiclesX(v) <= 35.0  & VehiclesY(v) < 6.0 & VehiclesY(v) > 2.0 & VehiclesVx(v) < 0.0))
  THEN
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL ||
    VehiclesY :(VehiclesY : Vehicles --> REAL & (Crash = TRUE or (VehiclesY(EgoVehicle) >= VehiclesY$0(EgoVehicle) - 0.05))) ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL
  END;

  FASTER =
  SELECT
    EgoVehicle : dom(VehiclesVx) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > 0.0 & VehiclesX(v) < 45.0  & VehiclesY(v) < 3.5 & VehiclesY(v) > -3.5))
  THEN
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL ||
    VehiclesY :: Vehicles --> REAL ||
    VehiclesVx : (VehiclesVx : Vehicles --> REAL & (Crash = TRUE or VehiclesVx(EgoVehicle) >= VehiclesVx$0(EgoVehicle) - 0.05)) ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx : (VehiclesAx : Vehicles --> REAL & (Crash = TRUE or VehiclesAx(EgoVehicle) >= -0.05)) ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL
  END;

  SLOWER =
  SELECT
    EgoVehicle : dom(VehiclesVx) &
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
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL || 
    VehiclesY :: Vehicles --> REAL || 
    VehiclesVx : (VehiclesVx : Vehicles --> REAL & (Crash = TRUE or VehiclesVx(EgoVehicle) <= VehiclesVx$0(EgoVehicle) + 0.05)) ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx : (VehiclesAx : Vehicles --> REAL & (Crash = TRUE or VehiclesAx(EgoVehicle) <= 0.05)) ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL
  END

END