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
    VehiclesAx := PresentVehicles * {0.0} ||
    VehiclesAy := PresentVehicles * {0.0} ||
    Reward :: REAL

OPERATIONS
  IDLE =
  SELECT
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) & VehiclesX(v) < VehiclesX(EgoVehicle) + 35.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 3.5 & VehiclesY(v) > VehiclesY(EgoVehicle) - 3.5 & VehiclesVx(v) < VehiclesVx(EgoVehicle)))
  THEN
    LET OldVehiclesVx, OldVehiclesVy BE OldVehiclesVx = VehiclesVx & OldVehiclesVy = VehiclesVy IN
        Crash :: BOOL ||
        PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles);
        VehiclesX :: PresentVehicles --> REAL ||
        VehiclesY :: PresentVehicles --> REAL ||
        VehiclesVx :: PresentVehicles --> REAL ||
        VehiclesVy :: PresentVehicles --> REAL;
        VehiclesAx := {v,a | v : PresentVehicles & a = VehiclesVx(v) - OldVehiclesVx(v)} ||
        VehiclesAy := {v,a | v : PresentVehicles & a = VehiclesVy(v) - OldVehiclesVy(v)} ||
        Reward :: REAL
    END
  END;

  LANE_LEFT =
  SELECT
    EgoVehicle : dom(VehiclesY) &
    VehiclesY(EgoVehicle) > 1.0 &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 25.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) - 15.0  & VehiclesY(v) > VehiclesY(EgoVehicle) - 6.0 & VehiclesY(v) < VehiclesY(EgoVehicle) - 2.0 & VehiclesVx(v) > VehiclesVx(EgoVehicle))) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 15.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) + 15.0  & VehiclesY(v) > VehiclesY(EgoVehicle) - 6.0 & VehiclesY(v) < VehiclesY(EgoVehicle) - 2.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) + 15.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) + 35.0  & VehiclesY(v) > VehiclesY(EgoVehicle) - 6.0 & VehiclesY(v) < VehiclesY(EgoVehicle) - 2.0 & VehiclesVx(v) < VehiclesVx(EgoVehicle)))
  THEN
    LET OldVehiclesVx, OldVehiclesVy BE OldVehiclesVx = VehiclesVx & OldVehiclesVy = VehiclesVy IN
        Crash :: BOOL ||
        PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles);
        VehiclesX :: PresentVehicles --> REAL ||
        VehiclesY :: PresentVehicles --> REAL ||
        VehiclesVx :: PresentVehicles --> REAL ||
        VehiclesVy :: PresentVehicles --> REAL;
        VehiclesAx := {v,a | v : PresentVehicles & a = VehiclesVx(v) - OldVehiclesVx(v)} ||
        VehiclesAy := {v,a | v : PresentVehicles & a = VehiclesVy(v) - OldVehiclesVy(v)} ||
        Reward :: REAL
    END
  END;

  LANE_RIGHT =
  SELECT
    EgoVehicle : dom(VehiclesY) &
    VehiclesY(EgoVehicle) < 7.0 &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 25.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) - 15.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 6.0 & VehiclesY(v) > VehiclesY(EgoVehicle) + 2.0 & VehiclesVx(v) > VehiclesVx(EgoVehicle))) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 15.0 & VehiclesX(v) < VehiclesX(EgoVehicle) + 15.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 6.0 & VehiclesY(v) > VehiclesY(EgoVehicle) + 2.0)) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) + 15.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) + 35.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 6.0 & VehiclesY(v) > VehiclesY(EgoVehicle) + 2.0 & VehiclesVx(v) < VehiclesVx(EgoVehicle)))
  THEN
    LET OldVehiclesVx, OldVehiclesVy BE OldVehiclesVx = VehiclesVx & OldVehiclesVy = VehiclesVy IN
        Crash :: BOOL ||
        PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles);
        VehiclesX :: PresentVehicles --> REAL ||
        VehiclesY :: PresentVehicles --> REAL ||
        VehiclesVx :: PresentVehicles --> REAL ||
        VehiclesVy :: PresentVehicles --> REAL;
        VehiclesAx := {v,a | v : PresentVehicles & a = VehiclesVx(v) - OldVehiclesVx(v)} ||
        VehiclesAy := {v,a | v : PresentVehicles & a = VehiclesVy(v) - OldVehiclesVy(v)} ||
        Reward :: REAL
    END
  END;

  FASTER =
  SELECT
    EgoVehicle : dom(VehiclesVx) &
    not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) & VehiclesX(v) < VehiclesX(EgoVehicle) + 45.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 3.5 & VehiclesY(v) > VehiclesY(EgoVehicle) - 3.5))
  THEN
    LET OldVehiclesVx, OldVehiclesVy BE OldVehiclesVx = VehiclesVx & OldVehiclesVy = VehiclesVy IN
        Crash :: BOOL ||
        PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles);
        VehiclesX :: PresentVehicles --> REAL ||
        VehiclesY :: PresentVehicles --> REAL ||
        VehiclesVx :: PresentVehicles --> REAL ||
        VehiclesVy :: PresentVehicles --> REAL;
        VehiclesAx := {v,a | v : PresentVehicles & a = VehiclesVx(v) - OldVehiclesVx(v)} ||
        VehiclesAy := {v,a | v : PresentVehicles & a = VehiclesVy(v) - OldVehiclesVy(v)} ||
        Reward :: REAL
    END
  END;

  SLOWER =
  SELECT
    EgoVehicle : dom(VehiclesVx) &
    (#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) & VehiclesX(v) < VehiclesX(EgoVehicle) + 15.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 3.5 & VehiclesY(v) > VehiclesY(EgoVehicle) - 3.5 & VehiclesVx(v) < VehiclesVx(EgoVehicle))
    =>
    (not(
    VehiclesY(EgoVehicle) > 1.0 &
        not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 25.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) - 15.0  & VehiclesY(v) > VehiclesY(EgoVehicle) - 6.0 & VehiclesY(v) < VehiclesY(EgoVehicle) - 2.0 & VehiclesVx(v) > VehiclesVx(EgoVehicle))) &
        not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 15.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) + 15.0  & VehiclesY(v) > VehiclesY(EgoVehicle) - 6.0 & VehiclesY(v) < VehiclesY(EgoVehicle) - 2.0)) &
        not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) + 15.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) + 35.0  & VehiclesY(v) > VehiclesY(EgoVehicle) - 6.0 & VehiclesY(v) < VehiclesY(EgoVehicle) - 2.0 & VehiclesVx(v) < VehiclesVx(EgoVehicle)))
    ) &
    not(
      VehiclesY(EgoVehicle) < 7.0 &
        not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 25.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) - 15.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 6.0 & VehiclesY(v) > VehiclesY(EgoVehicle) + 2.0 & VehiclesVx(v) > VehiclesVx(EgoVehicle))) &
        not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) - 15.0 & VehiclesX(v) < VehiclesX(EgoVehicle) + 15.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 6.0 & VehiclesY(v) > VehiclesY(EgoVehicle) + 2.0)) &
        not(#v. (v : PresentVehicles \ {EgoVehicle} & VehiclesX(v) > VehiclesX(EgoVehicle) + 15.0 & VehiclesX(v) <= VehiclesX(EgoVehicle) + 35.0  & VehiclesY(v) < VehiclesY(EgoVehicle) + 6.0 & VehiclesY(v) > VehiclesY(EgoVehicle) + 2.0 & VehiclesVx(v) < VehiclesVx(EgoVehicle)))
    )))
  THEN
    LET OldVehiclesVx, OldVehiclesVy BE OldVehiclesVx = VehiclesVx & OldVehiclesVy = VehiclesVy IN
        Crash :: BOOL ||
        PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles);
        VehiclesX :: PresentVehicles --> REAL ||
        VehiclesY :: PresentVehicles --> REAL ||
        VehiclesVx :: PresentVehicles --> REAL ||
        VehiclesVy :: PresentVehicles --> REAL;
        VehiclesAx := {v,a | v : PresentVehicles & a = VehiclesVx(v) - OldVehiclesVx(v)} ||
        VehiclesAy := {v,a | v : PresentVehicles & a = VehiclesVy(v) - OldVehiclesVy(v)} ||
        Reward :: REAL
    END
  END

END