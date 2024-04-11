MACHINE HighwayEnvironment_RSS_v2

SETS
    Vehicles
DEFINITIONS
    VISB_JSON_FILE == "visualization/HighwayEnvRSS_v2.json";
    ANIMATION_EXPRESSION_dRSS_cur_lane == dRSS_cur_lane;
    ANIMATION_EXPRESSION_dRSS_left_lane == dRSS_left_lane;
    ANIMATION_EXPRESSION_dRSS_right_lane == dRSS_right_lane;
    //SET_PREF_MAX_OPERATIONS == 0; // comment in to check LTL formulas below on replayed traces only; comment out to apply Real-Time simulation
    // check that environmental assumptions are true
    ASSERT_LTL1 == "G(BA({VehiclesVx$0(EgoVehicle) <= VehiclesVx(EgoVehicle)+aMax}))";
    deltavSV == VehiclesVx(EgoVehicle)-VehiclesVx$0(EgoVehicle);
    ASSERT_LTL2 == "G(BA({!v.(v:OtherV /\ PresentVehicles$0 => (VehiclesVx$0(v)+VehiclesVx$0(EgoVehicle)+aMax >= VehiclesVx(v)+VehiclesVx(EgoVehicle)))}))";
    ASSERT_LTL3 == "G(BA({VehiclesVx$0(EgoVehicle) >= VehiclesVx(EgoVehicle)-aBrakeMax}))";
    ASSERT_LTL4 == "G(BA({!v.(v:OtherV /\ PresentVehicles$0 => (VehiclesVx$0(v)+VehiclesVx$0(EgoVehicle)-aBrakeMax <= VehiclesVx(v)+VehiclesVx(EgoVehicle)))}))";
    ANIMATION_EXPRESSION_rspeed == %v.(v:OtherV|VehiclesVx(v)+VehiclesVx(EgoVehicle));
    brakeDistance(initialSpeed,brakeAcc) == (initialSpeed**2)/(2.0*brakeAcc);
    // and now the distance required to brake until a target speed which is not necessarily 0.0:
    brakeDistanceWithTarget(initialSpeed,targetSpeed,brakeAcc) == (initialSpeed**2 - targetSpeed**2)/(2.0*brakeAcc);
    minAbsoluteSpeed == 13.5; // minimum speed of other vehicles
    minEgoSpeed == 19.5; // minimum speed of Ego vehicle
    brakeDistance20(initialSpeed,brakeAcc) == brakeDistanceWithTarget(initialSpeed,minAbsoluteSpeed,brakeAcc);
    
    distanceTravelled(deltaTime,initialSpeed,acceleration) ==
      IF initialSpeed < -(acceleration*deltaTime)
      THEN brakeDistance(initialSpeed,-acceleration) // we reach speed 0 and then stop
      ELSE initialSpeed*deltaTime + 0.5*acceleration*(deltaTime**2)
      END;
    vSV == VehiclesVx(EgoVehicle);
    minOtherSpeed == min({v•v:OtherV|vSV+VehiclesVx(v)});
    reactionDistance == distanceTravelled(reactionTime,vSV,aMax)
     /* "distance car travels during reaction time supposing maximal acceleration" */;

    dRSS_wo_max_absolute(maxAccRear,vFront) ==
       distanceTravelled(reactionTime,vSV,maxAccRear) +
       brakeDistance(vSV+maxAccRear*reactionTime,aBrakeMin) -
       brakeDistance(vFront,aBrakeMax);

    // now a version just taking relative speed to minAbsoluteSpeed into account:
    dRSS_wo_max(maxAccRear,vFront) ==
       distanceTravelled(reactionTime,vSV,maxAccRear) +
       brakeDistance(vSV+maxAccRear*reactionTime,aBrakeMin) -
       brakeDistance(vFront,aBrakeMax);
    OtherV == PresentVehicles \ {EgoVehicle};
    LeftLane == -1;
    CurLane == 0;
    RightLane == 1;
    // TODO: also take Vy into account:
    GetLane(vv) == floor(VehiclesY(vv)/4.0+0.5);
    MaxOneLane(Delta) == IF Delta < -4.0 THEN -4.0 ELSIF Delta>4.0 THEN 4.0 ELSE Delta END;
    GetNextLane(vv) == floor((VehiclesY(vv)+MaxOneLane(VehiclesVy(vv)*reactionTime))/4.0+0.5); // predicted lane after one cycle; assume at most one lane change (4 metres)
    VehiclesInSameLane == {v|v:OtherV & GetLane(v)=CurLane};
    VehiclesInLeftLane == {v|v:OtherV & GetLane(v)=LeftLane};
    VehiclesInRightLane == {v|v:OtherV & GetLane(v)=RightLane};
    VehiclesInLane(Lane) == {v|v:OtherV & (GetLane(v)=Lane or GetNextLane(v)=Lane)};
    NextVehiclesInLane(Lane) == {v|v:VehiclesInLane(Lane) & VehiclesX(v) = min(VehiclesX[VehiclesInLane(Lane)])};
    // TODO: we should discard those vehicles that are sufficiently behind Ego
    // e.g., such that their position + their dRSS distance wrt Ego is ok
  
    dRSS_lane(maxAccRear,Lane) == max({0.0, 
                       dRSS_wo_max(maxAccRear,vSV+min({30.0} \/ VehiclesVx[NextVehiclesInLane(Lane)]))});
    dRSS_cur_lane == dRSS_lane(aMax,CurLane);
    dRSS_left_lane == dRSS_lane(0.05,LeftLane); // if we go left we will accelerate by at most 0.05 (TODO:check)
    dRSS_right_lane == dRSS_lane(0.05,RightLane); // ditto
    
    FASTER_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(aMax,CurLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(CurLane)]));
    IDLE_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(0.05,CurLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(CurLane)]));
    aBrakeMinCorrected == IF vSV <= minEgoSpeed THEN 0.0
                             ELSIF vSV > minEgoSpeed + aBrakeMin THEN aBrakeMin
                             ELSE vSV - minEgoSpeed END; // corrected aBrakeMin taking into account Ego vehicle does not
    SLOWER_ALLOWED == (NextVehiclesInLane(CurLane)/={}
                         => dRSS_lane(-aBrakeMin,CurLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(CurLane)])
                             // TODO: take minimal speed into account
                     );
    COLLISION_DANGER_CUR_LANE(otherLane,margin) == // collision danger in current lane, otherLane has better chances
                      (NextVehiclesInLane(CurLane)/={} &
                          LET overshoot 
                          BE overshoot = dRSS_lane(-aBrakeMinCorrected,CurLane) + vehLen
                                         - min(VehiclesX[NextVehiclesInLane(CurLane)])
                          IN overshoot > 0.0 /*@desc "staying in current lane is not safe" */ &
                            // TODO: check that switching to other lane is also not safe !
                             overshoot - margin > dRSS_lane(0.05,otherLane) + vehLen
                                                  - min(VehiclesX[NextVehiclesInLane(otherLane)])
                          END
                     );
    LEFT_ALLOWED == VehiclesY(EgoVehicle) > 1.0 & 
                    (NextVehiclesInLane(LeftLane)/={}
                         => dRSS_lane(0.05,LeftLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(LeftLane)])
                    );
    LEFT_ALLOWED_OR_IMPROVES_CHANCES(margin) == VehiclesY(EgoVehicle) > 1.0 & 
                    (NextVehiclesInLane(LeftLane)/={}
                         => (dRSS_lane(0.05,LeftLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(LeftLane)])
                            or
                            COLLISION_DANGER_CUR_LANE(LeftLane,margin)
                            )
                    );
    RIGHT_ALLOWED == VehiclesY(EgoVehicle) < 7.0 &
                     (NextVehiclesInLane(RightLane)/={}
                         => dRSS_lane(0.05,RightLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(RightLane)])
                    );
    RIGHT_ALLOWED_OR_IMPROVES_CHANCES(margin) == VehiclesY(EgoVehicle) < 7.0 &
                     (NextVehiclesInLane(RightLane)/={}
                         => (dRSS_lane(0.05,RightLane) + vehLen <= min(VehiclesX[NextVehiclesInLane(RightLane)])
                            or
                            COLLISION_DANGER_CUR_LANE(RightLane,margin))
                    )
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
    IDLE_ALLOWED
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
    (LEFT_ALLOWED or
    (not(RIGHT_ALLOWED) & not(FASTER_ALLOWED) & not(IDLE_ALLOWED) & not(SLOWER_ALLOWED) & LEFT_ALLOWED_OR_IMPROVES_CHANCES(0.0)))
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

  LANE_RIGHT =
  SELECT
    EgoVehicle : dom(VehiclesY) &
    (RIGHT_ALLOWED or
    (not(LEFT_ALLOWED) & not(FASTER_ALLOWED) & not(IDLE_ALLOWED) & not(SLOWER_ALLOWED) & RIGHT_ALLOWED_OR_IMPROVES_CHANCES(0.0)))
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

  FASTER =
  SELECT
    EgoVehicle : dom(VehiclesVx) &
    FASTER_ALLOWED
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

  SLOWER =
  SELECT
    EgoVehicle : dom(VehiclesVx) &
    (not(SLOWER_ALLOWED)
    =>
    (not(LEFT_ALLOWED_OR_IMPROVES_CHANCES(15.0)) & 
     not(RIGHT_ALLOWED_OR_IMPROVES_CHANCES(15.0))))
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
  END

END