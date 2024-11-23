/*
Developed by Michael Leuschel
*/
MACHINE HighwayEnvironment_RSS_v2

SETS
    Vehicles
DEFINITIONS
    VISB_JSON_FILE == "visualization/HighwayEnvRSS_v2.json";
    brakeDistance(initialSpeed,brakeAcc) == (initialSpeed**2)/(2.0*brakeAcc);
    // and now the distance required to brake until a target speed which is not necessarily 0.0:
    brakeDistanceWithTarget(initialSpeed,targetSpeed,brakeAcc) == (initialSpeed**2 - targetSpeed**2)/(2.0*brakeAcc);
    minAbsoluteSpeed == 13.5;
    brakeDistance20(initialSpeed,brakeAcc) == brakeDistanceWithTarget(initialSpeed,minAbsoluteSpeed,brakeAcc);
    distanceTravelled(deltaTime,initialSpeed,acceleration) ==
      IF initialSpeed < -(acceleration*deltaTime)
      THEN brakeDistance(initialSpeed,-acceleration) // we reach speed 0 and then stop
      ELSE initialSpeed*deltaTime + 0.5*acceleration*(deltaTime**2)
      END;
    vSV == VehiclesVx(EgoVehicle);
    minOtherSpeed == min({v•v:OtherV|VehiclesVx(v)});
    reactionDistance == distanceTravelled(reactionTime,vSV,aMax)
     /* "distance car travels during reaction time supposing maximal acceleration" */;

    dRSS_wo_max_absolute(maxAccRear,vFront) ==
       distanceTravelled(reactionTime,vSV,maxAccRear) +
       brakeDistance(vSV+maxAccRear*reactionTime,aBrakeMin) -
       brakeDistance(vFront,aBrakeMax);

    // now a version just taking relative speed to minAbsoluteSpeed into account:
    dRSS_wo_max(maxAccRear,vFront) ==
       distanceTravelled(reactionTime,vSV-minAbsoluteSpeed,maxAccRear) +
       brakeDistance(vSV-minAbsoluteSpeed+maxAccRear*reactionTime,aBrakeMin) -
       brakeDistance(vFront-minAbsoluteSpeed,aBrakeMax);
    OtherV == Vehicles \ {EgoVehicle};
    LeftLane == -1;
    CurLane == 0;
    RightLane == 1;
    // TODO: also take Vy into account:

    VehiclesX_relative == {x,y | x : Vehicles & y = VehiclesX(x) - VehiclesX(EgoVehicle)};
    VehiclesY_relative == {x,y | x : Vehicles & y = VehiclesY(x) - VehiclesY(EgoVehicle)};
    VehiclesVx_relative == {x,y | x : Vehicles & y = VehiclesVx(x) - VehiclesVx(EgoVehicle)};
    VehiclesVy_relative == {x,y | x : Vehicles & y = VehiclesVy(x) - VehiclesVy(EgoVehicle)};
    GetLane(vv) == floor(VehiclesY_relative(vv)/4.0+0.5);
    GetNextLane(vv) == floor((VehiclesY_relative(vv)+VehiclesVy(vv)*reactionTime)/4.0+0.5); // predicated lane after one cycle
    VehiclesInSameLane == {v|v:OtherV & GetLane(v)=CurLane};
    VehiclesInLeftLane == {v|v:OtherV & GetLane(v)=LeftLane};
    VehiclesInRightLane == {v|v:OtherV & GetLane(v)=RightLane};
    VehiclesInLane(Lane) == {v|v:OtherV & (GetLane(v)=Lane or GetNextLane(v)=Lane)};


    NextVehiclesInLane(Lane) == {v|v:VehiclesInLane(Lane) & VehiclesX_relative(v) = min(VehiclesX_relative[VehiclesInLane(Lane)])};

    dRSS_lane(maxAccRear,Lane) == max({0.0,
                       dRSS_wo_max(maxAccRear,vSV+min({30.0} \/ VehiclesVx_relative[NextVehiclesInLane(Lane)]))});
    dRSS_cur_lane == dRSS_lane(aMax,CurLane);
    dRSS_left_lane == dRSS_lane(0.05,LeftLane); // if we go left we will accelerate by at most 0.05 (TODO:check)
    dRSS_right_lane == dRSS_lane(0.05,RightLane); // ditto

    FASTER_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(aMax,CurLane) + vehLen <= min(VehiclesX_relative[NextVehiclesInLane(CurLane)]));
    IDLE_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(0.05,CurLane) + vehLen <= min(VehiclesX_relative[NextVehiclesInLane(CurLane)]));
    SLOWER_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(0.05,CurLane) + vehLen <= min(VehiclesX_relative[NextVehiclesInLane(CurLane)]));
    LEFT_ALLOWED == VehiclesY(EgoVehicle) > 1.0 &
                    (NextVehiclesInLane(LeftLane)/={}
                         => dRSS_lane(0.05,LeftLane) + vehLen <= min(VehiclesX_relative[NextVehiclesInLane(LeftLane)]));
    RIGHT_ALLOWED == VehiclesY(EgoVehicle) < 7.0 &
                     (NextVehiclesInLane(RightLane)/={}
                         => dRSS_lane(0.05,RightLane) + vehLen <= min(VehiclesX_relative[NextVehiclesInLane(RightLane)]))
ASSERTIONS
    dRSS_cur_lane >= 0.0;
    dRSS_left_lane >= 0.0;
    dRSS_right_lane >= 0.0
CONSTANTS
    aMax         /*@desc "maximum acceleration of vehicles" */,
    aBrakeMax    /*@desc "maximum braking deceleration of vehicles" */,
    aBrakeMin    /*@desc "guaranteed minimum braking deceleration of vehicles" */,
    vehLen       /*@desc "vehicle length" */,
    reactionTime /*@desc "max. time driver needs to react, i.e., controller cycle time" */,
    EgoVehicle   /*@desc "The main vehicle being controlled" */
PROPERTIES
    card(Vehicles) > 1 &
    card(Vehicles) = 5 &
    EgoVehicle : Vehicles &
    
    {aMax,aBrakeMax,aBrakeMin,reactionTime} <: REAL &
    aMax = 5.0 &
    aBrakeMax = 5.1 &
    aBrakeMin = 4.9 &
    vehLen = 5.0 &
    reactionTime = 1.0

VARIABLES
    Crash       /*@desc "TRUE when a crash occurred" */,
    PresentVehicles /*@desc "The vehicles that are on screen" */,
    VehiclesX   /*@desc "x-coordinates of the vehicles" */,
    VehiclesY   /*@desc "y-coordinates of the vehicles" */,
    VehiclesVx  /*@desc "horizontal speed of the vehicles" */,
    VehiclesVy  /*@desc "vertical speed of the vehicles" */,
    VehiclesAx  /*@desc "horizontal acceleration of the vehicles" */,
    VehiclesAy  /*@desc "vertical acceleration of the vehicles" */,
    Reward

INVARIANT
    Crash : BOOL &
    PresentVehicles : POW(Vehicles) &
    EgoVehicle : PresentVehicles /*@desc "the Ego vehicle is always present" */ &
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
    IDLE_ALLOWED
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
    LEFT_ALLOWED
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
    RIGHT_ALLOWED
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
    FASTER_ALLOWED
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
    (not(SLOWER_ALLOWED)
    =>
    (not(LEFT_ALLOWED) &
    not(RIGHT_ALLOWED)))
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