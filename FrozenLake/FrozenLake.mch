/*
Developed by Fabian Vu
*/
MACHINE FrozenLake

SETS 
  GROUND = {ICE, HOLE}

VARIABLES 
  x, 
  y, 
  goal_x, 
  goal_y,
  field

CONSTANTS n
PROPERTIES n=8

INVARIANT 
  x : INTEGER & 
  y : INTEGER & 
  goal_x : INTEGER & 
  goal_y : INTEGER &
  field : ((0..n-1) * (0..n-1)) --> GROUND


INITIALISATION
  x :: INTEGER ||
  y :: INTEGER ||
  goal_x :: INTEGER ||
  goal_y :: INTEGER ||
  field :: ((0..n-1) * (0..n-1)) --> GROUND

OPERATIONS
  Move_Left = 
  PRE
    x > 0 & field(x - 1 |-> y) = ICE
  THEN
    x := x - 1
  END;

  Move_Down = 
  PRE
    y < n-1 & field(x |-> y + 1) = ICE
  THEN
    y := y + 1
  END;

  Move_Right = 
  PRE
    x < n-1 & field(x + 1 |-> y) = ICE
  THEN
    x := x + 1
  END;

  Move_Up = 
  PRE
    y > 0 & field(x |-> y - 1) = ICE
  THEN
    y := y - 1
  END

END