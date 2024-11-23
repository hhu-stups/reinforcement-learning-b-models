/*
Developed by Fabian Vu
*/
MACHINE CliffWalking

SETS 
  GROUND = {GRASS, CLIFF}

VARIABLES 
  x, 
  y, 
  goal_x, 
  goal_y,
  field

CONSTANTS m, n, cliffs
PROPERTIES m = 4 & n = 12 & cliffs = (1..10) * {3}

INVARIANT 
  x : INTEGER & 
  y : INTEGER & 
  goal_x : INTEGER & 
  goal_y : INTEGER &
  field : ((0..n-1) * (0..m-1)) --> GROUND


INITIALISATION
  x :: INTEGER ||
  y :: INTEGER ||
  goal_x := 11 ||
  goal_y := 3 ||
  field := (((0..n-1) * (0..m-1)) * {GRASS}) <+ {a,b | a : cliffs & b = CLIFF}

OPERATIONS
  Move_Left = 
  PRE
    x > 0 & field(x - 1 |-> y) = GRASS
  THEN
    x := x - 1
  END;

  Move_Down = 
  PRE
    y < m-1 & field(x |-> y + 1) = GRASS
  THEN
    y := y + 1
  END;

  Move_Right = 
  PRE
    x < n-1 & field(x + 1 |-> y) = GRASS
  THEN
    x := x + 1
  END;

  Move_Up = 
  PRE
    y > 0 & field(x |-> y - 1) = GRASS
  THEN
    y := y - 1
  END

END