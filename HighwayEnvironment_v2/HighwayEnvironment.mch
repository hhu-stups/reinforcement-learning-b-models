MACHINE HighwayEnvironment

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
  IDLE = BEGIN
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

  LANE_LEFT = BEGIN
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

  LANE_RIGHT = BEGIN
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

  FASTER = BEGIN
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

  SLOWER = BEGIN
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