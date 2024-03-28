MACHINE HighwayEnvironment_RSS

SETS
    Vehicles
DEFINITIONS
    VISB_JSON_FILE == "visualization/HighwayEnvRSS.json";
    brakeDistance(initialSpeed,brakeAcc) == (initialSpeed**2)/(2.0*brakeAcc);
    distanceTravelled(deltaTime,initialSpeed,acceleration) ==
      IF initialSpeed < -(acceleration*deltaTime)
      THEN brakeDistance(initialSpeed,-acceleration) // we reach speed 0 and then stop
      ELSE initialSpeed*deltaTime + 0.5*acceleration*(deltaTime**2)
      END;
    vSV == VehiclesVx(EgoVehicle);
    reactionDistance == distanceTravelled(reactionTime,vSV,aMax)
     /* "distance car travels during reaction time supposing maximal acceleration" */;
     
    dRSS_wo_max(maxAccRear,vFront) ==
       distanceTravelled(reactionTime,vSV,maxAccRear) +
       brakeDistance(vSV+maxAccRear*reactionTime,aBrakeMin) -
       brakeDistance(vFront,aBrakeMax);
    OtherV == Vehicles \ {EgoVehicle};
    LeftLane == -2;
    CurLane == 0;
    RightLane == 2;
    VehiclesInSameLane == {v|v:OtherV & floor(VehiclesY(v)/2.0+0.5)=CurLane};
    VehiclesInLeftLane == {v|v:OtherV & floor(VehiclesY(v)/2.0+0.5)=LeftLane};
    VehiclesInRightLane == {v|v:OtherV & floor(VehiclesY(v)/2.0+0.5)=RightLane};
    VehiclesInLane(Lane) == {v|v:OtherV & floor(VehiclesY(v)/2.0+0.5)=Lane};
    NextVehiclesInLane(Lane) == {v|v:VehiclesInLane(Lane) & VehiclesX(v) = min(VehiclesX[VehiclesInLane(Lane)])};
  
    dRSS_lane(maxAccRear,Lane) == max({0.0, 
                       dRSS_wo_max(maxAccRear,vSV+min({30.0} \/ VehiclesVx[NextVehiclesInLane(Lane)]))});
    dRSS_cur_lane == dRSS_lane(aMax,CurLane);
    dRSS_left_lane == dRSS_lane(0.05,LeftLane); // if we go left we will accelerate by at most 0.05 (TODO:check)
    dRSS_right_lane == dRSS_lane(0.05,RightLane); // ditto
    
    FASTER_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(aMax,CurLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(CurLane)]));
    IDLE_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(0.05,CurLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(CurLane)]));
    LEFT_ALLOWED == VehiclesY(EgoVehicle) > 1.0 & 
                    (NextVehiclesInLane(LeftLane)/={}
                         => dRSS_lane(0.05,LeftLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(LeftLane)]));
    RIGHT_ALLOWED == VehiclesY(EgoVehicle) < 7.0 &
                     (NextVehiclesInLane(RightLane)/={}
                         => dRSS_lane(0.05,RightLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(RightLane)]))
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
    Reward,
    ErrorMsg

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
    Reward : REAL &
    ErrorMsg : STRING

INITIALISATION
    Crash :: BOOL ||
    PresentVehicles : (PresentVehicles : POW(Vehicles) & EgoVehicle : PresentVehicles) ||
    VehiclesX :: Vehicles --> REAL ||
    VehiclesY :: Vehicles --> REAL ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL ||
    ErrorMsg := ""

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
    Reward :: REAL ||
    IF not(IDLE_ALLOWED) THEN ErrorMsg := ```IDLE was not ALLOWED at x:$[2f]{VehiclesX(EgoVehicle)} y:$[2f]{VehiclesY(EgoVehicle)}``` ELSE ErrorMsg := "IDLE ok" END
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
    VehiclesY :: Vehicles --> REAL ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL ||
    IF not(LEFT_ALLOWED) THEN ErrorMsg := ```LEFT was not ALLOWED at x:$[2f]{VehiclesX(EgoVehicle)} y:$[2f]{VehiclesY(EgoVehicle)}``` ELSE ErrorMsg := "LEFT ok" END
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
    VehiclesY :: Vehicles --> REAL ||
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL ||
    IF not(RIGHT_ALLOWED) THEN ErrorMsg := ```RIGHT was not ALLOWED at x:$[2f]{VehiclesX(EgoVehicle)} y:$[2f]{VehiclesY(EgoVehicle)}``` ELSE ErrorMsg := "RIGHT ok" END
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
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL ||
    IF not(FASTER_ALLOWED) THEN ErrorMsg := ```FASTER was not ALLOWED at x:$[2f]{VehiclesX(EgoVehicle)} y:$[2f]{VehiclesY(EgoVehicle)}``` ELSE ErrorMsg := "FASTER ok" END
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
    VehiclesVx :: Vehicles --> REAL ||
    VehiclesVy :: Vehicles --> REAL ||
    VehiclesAx :: Vehicles --> REAL ||
    VehiclesAy :: Vehicles --> REAL ||
    Reward :: REAL ||
    ErrorMsg := "SLOWER (ok)"
  END

END