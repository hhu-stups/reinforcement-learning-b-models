MACHINE AutonomousDroneSwarm_abstract

VARIABLES 
  visited,
  drone_positions,
  current_drone

CONSTANTS m, n
PROPERTIES m = 10 & n = 2

INVARIANT
  visited <: ((0..m-1) * (0..m-1)) &
  drone_positions : 1..n --> (0..m-1) * (0..m-1) &
  current_drone : 1..n


INITIALISATION
  drone_positions :: 1..n --> (0..m-1) * (0..m-1);
  visited := ran(drone_positions);
  current_drone := 1

OPERATIONS
  Move_Left =
  PRE
    prj1(drone_positions(current_drone)) > 0 & 
    prj1(drone_positions(current_drone)) - 1 |-> prj2(drone_positions(current_drone)) /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) - 1 |-> prj2(drone_positions(current_drone));
    visited := visited \/ {drone_positions(current_drone)};
    current_drone := IF current_drone + 1 = n+1 THEN 1 ELSE current_drone + 1 END
  END;

  Move_Down =
  PRE
    prj2(drone_positions(current_drone)) < m-1 &
    prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) + 1 /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) + 1;
    visited := visited \/ {drone_positions(current_drone)};
    current_drone := IF current_drone + 1 = n+1 THEN 1 ELSE current_drone + 1 END
  END;

  Move_Right =
  PRE
    prj1(drone_positions(current_drone)) < m-1 &
    prj1(drone_positions(current_drone)) + 1 |-> prj2(drone_positions(current_drone)) /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) + 1 |-> prj2(drone_positions(current_drone));
    visited := visited \/ {drone_positions(current_drone)};
    current_drone := IF current_drone + 1 = n+1 THEN 1 ELSE current_drone + 1 END
  END;

  Move_Up =
  PRE
    prj2(drone_positions(current_drone)) > 0 &
    prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) - 1 /: ran(drone_positions)
  THEN
    drone_positions(current_drone) := prj1(drone_positions(current_drone)) |-> prj2(drone_positions(current_drone)) - 1;
    visited := visited \/ {drone_positions(current_drone)};
    current_drone := IF current_drone + 1 = n+1 THEN 1 ELSE current_drone + 1 END
  END


END