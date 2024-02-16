MACHINE AutonomousDroneSwarm

VARIABLES 
  visited,
  drone_positions

CONSTANTS m, n
PROPERTIES m = 10 & n = 2

INVARIANT
  visited : POW((1..n) * ((0..m-1) * (0..m-1))) &
  drone_positions : 1..n --> (0..m-1) * (0..m-1) &
  card(ran(drone_positions)) <= 2


INITIALISATION
  drone_positions :: 1..n --> (0..m-1) * (0..m-1);
  visited := drone_positions

OPERATIONS
  Move_Left(drone) =
  PRE
    drone : 1.. n & prj1(drone_positions(drone)) > 0/* & 
    prj1(drone_positions(drone)) - 1 |-> prj2(drone_positions(drone)) /: ran(drone_positions)*/
  THEN
    drone_positions :: 1..n --> (0..m-1) * (0..m-1);
    visited := visited \/ (drone_positions |>> ran(visited))
  END;

  Move_Down(drone) =
  PRE
    drone : 1.. n & prj2(drone_positions(drone)) < m/* &
    prj1(drone_positions(drone)) |-> prj2(drone_positions(drone)) + 1 /: ran(drone_positions)*/
  THEN
    drone_positions :: 1..n --> (0..m-1) * (0..m-1);
    visited := visited \/ (drone_positions |>> ran(visited))
  END;

  Move_Right(drone) =
  PRE
    drone : 1.. n & prj1(drone_positions(drone)) < m/* &
    prj1(drone_positions(drone)) + 1 |-> prj2(drone_positions(drone)) /: ran(drone_positions)*/
  THEN
    drone_positions :: 1..n --> (0..m-1) * (0..m-1);
    visited := visited \/ (drone_positions |>> ran(visited))
  END;

  Move_Up(drone) =
  PRE
    drone : 1.. n & prj2(drone_positions(drone)) > 0/* &
    prj1(drone_positions(drone)) |-> prj2(drone_positions(drone)) - 1 /: ran(drone_positions)*/
  THEN
    drone_positions :: 1..n --> (0..m-1) * (0..m-1);
    visited := visited \/ (drone_positions |>> ran(visited))
  END


END