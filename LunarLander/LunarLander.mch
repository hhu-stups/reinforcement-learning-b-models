/*
Developed by Fabian Vu
*/
MACHINE LunarLander

VARIABLES  x, y, v_x, v_y, angle, v_angular, left_leg_on_ground, right_leg_on_ground, surface

INVARIANT x : REAL & y : REAL & v_x : REAL & v_y : REAL &
          angle : REAL & v_angular : REAL &
          left_leg_on_ground : BOOL & right_leg_on_ground : BOOL &
          surface : 1..10 --> ((REAL * REAL) * (REAL * REAL))

CONSTANTS c
PROPERTIES c=1

INITIALISATION
  x :: REAL ||
  y :: REAL ||
  v_x :: REAL ||
  v_y :: REAL ||
  angle :: REAL ||
  v_angular :: REAL ||
  left_leg_on_ground :: BOOL ||
  right_leg_on_ground :: BOOL ||
  surface :: 1..10 --> ((REAL * REAL) * (REAL * REAL))

OPERATIONS
  Do_Nothing = BEGIN
    x :: REAL ||
    y :: REAL ||
    v_x :: REAL ||
    v_y :: REAL ||
    angle :: REAL ||
    v_angular :: REAL ||
    left_leg_on_ground :: BOOL ||
    right_leg_on_ground :: BOOL ||
    surface :: 1..10 --> ((REAL * REAL) * (REAL * REAL))
  END;

  Fire_Left = BEGIN
    x :: REAL ||
    y :: REAL ||
    v_x :: REAL ||
    v_y :: REAL ||
    angle :: REAL ||
    v_angular :: REAL ||
    left_leg_on_ground :: BOOL ||
    right_leg_on_ground :: BOOL ||
    surface :: 1..10 --> ((REAL * REAL) * (REAL * REAL))
  END;

  Fire_Right = BEGIN
    x :: REAL ||
    y :: REAL ||
    v_x :: REAL ||
    v_y :: REAL ||
    angle :: REAL ||
    v_angular :: REAL ||
    left_leg_on_ground :: BOOL ||
    right_leg_on_ground :: BOOL ||
    surface :: 1..10 --> ((REAL * REAL) * (REAL * REAL))
  END;

  Fire_Down = BEGIN
    x :: REAL ||
    y :: REAL ||
    v_x :: REAL ||
    v_y :: REAL ||
    angle :: REAL ||
    v_angular :: REAL ||
    left_leg_on_ground :: BOOL ||
    right_leg_on_ground :: BOOL ||
    surface :: 1..10 --> ((REAL * REAL) * (REAL * REAL))
  END

END