/* ********************************************************************
 * Copyright (c) 2021 Biological Modeling and Visualization Laboratory,
 * University of Calgary. All rights reserved.
 * ********************************************************************/

/* Descriptive model of maize */
#include <array>
#include <cmath>
#include <lpfgall.h>

#define STEPS 1000

/* Declaration of constants */
const float DT = 1.f;
const float DAGE = 1.f; // apex and leaf age increment
const float DX = 0.05f; // step size for drawing a leaf as generalized cylinder

/* Mapping of functions from functions.fset to macros */

#define br_angle(t) tfunc(BR_ANGLE,t)
#define br_target_angle(n) func(BR_TARGET_ANGLE,n/val(MAX_PHYTOMERS))

#define leaf_length(t)	tfunc(LEAF_LENGTH,t)
#define leaf_target_len(n) func(LEAF_TARGET_LEN,n/val(MAX_PHYTOMERS))
#define leaf_width(x)	func(LEAF_WIDTH,float(x))
#define leaf_bend(x)	(val(LEAF_BEND_SCALE)*func(LEAF_BEND,float(x)))
#define leaf_twist(x)	(val(LEAF_TWIST_SCALE)*func(LEAF_TWIST,float(x)))

#define internode_length(t) tfunc(INTERNODE_LENGTH,t)
#define internode_target_len(n) func(INTERNODE_TARGET_LEN,n/val(MAX_PHYTOMERS))
#define internode_width(n) func(INTERNODE_WIDTH,n/val(MAX_PHYTOMERS))

/* Structure declarations and initialization */

typedef std::array<float,10> std_array;
std_array initArray = {0,0,0,0,0,0,0,0,0,0};

void UpdateAges (float sim_time, float age, std_array &ages) {
    if (sim_time <= 1.)
        ages[0] = age;
    else if (sim_time <= 3.)
        ages[1] = age;
    else if (sim_time <= 6.)
        ages[2] = age;
    else if (sim_time <= 9.)
        ages[3] = age;
    else if (sim_time <= 12.)
        ages[4] = age;
    else if (sim_time <= 15.)
        ages[5] = age;
    else if (sim_time <= 18.)
        ages[6] = age;
    else if (sim_time <= 21.)
        ages[7] = age;
    else if (sim_time <= 24.)
        ages[8] = age;
    else if (sim_time <= 27.)
        ages[9] = age;
}

/* Module definitions */

module A(float,float,std_array); // node number, age, array of ages at specific times
module I(float,float,std_array);
module L(float,float,std_array);
module C(float,float); // leaf blade
module VisualAid();

/* Global variables */

float sim_time = 0.;

/* Organization of computation into phases */

Start: {
    sim_time = 0.;
    UseView(view001);
    UseView(view003);
    UseView(view006);
    UseView(view009);
    UseView(view012);
    UseView(view015);
    UseView(view018);
    UseView(view021);
    UseView(view024);
    UseView(view027);
}

EndEach: {
    sim_time += DT;
}

/* Start of L-system */

derivation length: STEPS;

Axiom: SB Down(4.) RollL(90.) A(val(PLASTOCHRON),0,initArray) EB VisualAid;

production:

A(age,n,ages) : {
    age += DAGE*DT;
    UpdateAges(sim_time,age,ages);
	produce A(age,n,ages);
}

L(age,n,ages) : {
    age += DAGE*DT;
    UpdateAges(sim_time,age,ages);
	produce L(age,n,ages);
}

I(age,n,ages) : {
    age += DAGE*DT;
    UpdateAges(sim_time,age,ages);
    produce I(age,n,ages);
}

decomposition:
maximum depth: 100;

A(age,n,ages) : {
	if (age >= val(PLASTOCHRON) && n < val(MAX_PHYTOMERS)) {
        age = age - val(PLASTOCHRON); // decrement apex age
		produce I(age,n,initArray) // create new internode
			SB  // start branch
                // branching angle
                Down(val(BrAngle) * br_target_angle(n))
                L(age,n,initArray) // create leaf at position n
            EB // end branch
			RollL(180.0) // distichous phyllotaxis
			A(age,n+1,ages); // recreate apex
	}
}

interpretation:
maximum depth: 100;

// define the interpretation rules as macros so they
// can be reused in each vgroup

#define InternodeInterpretation(agesIndex) \
I(age,n,ages): { \
    if (ages[agesIndex] <= 0.) \
        produce; \
    float len = val(IntLen) * internode_target_len(n) * internode_length(ages[agesIndex]); \
    float wid = std::powf(len,val(ExpIntRad)) * val(IntWid) * internode_width(n); \
    int colour = (int(n) % 3) + 2; \
    produce SetColor(colour) SetWidth(wid) F(len); \
} 

#define LeafInterpretation(agesIndex) \
L(age,n,ages): { \
    if (ages[agesIndex] <= 0.) \
        produce; \
	float len = val(LeafLen) * leaf_target_len(n) * leaf_length(ages[agesIndex]); \
    nproduce SetWidth(std::powf(len,val(ExpLeafWid)) * val(LeafWid) * leaf_width(0)); \
	nproduce Down(br_angle(ages[agesIndex])) StartGC(); \
        float x = 0.f; \
        for (x = 0.; x < 1.; x += DX) { \
            nproduce BlendedContour(0,1,std::pow(x,0.2)); \
            nproduce F(len*DX); \
		    nproduce SetWidth(std::powf(len,val(ExpLeafWid)) * val(LeafWid) * leaf_width(x)) \
                    Down(leaf_bend(x)*len*DX) \
                    RollL(leaf_twist(x)*len*DX); \
        } \
    nproduce F(len*(1.-x)); \
    produce EndGC(); \
} 

#define ViualAidInterpretation(day) \
VisualAid() : { \
    const int MAX_DAYS = 27; \
    static char str[64] = "Day 1"; \
	const float SF = 0.01; \
	float w = 1227.0 * SF; \
	float h = 1028.0 * SF; \
    nproduce SetElasticity(0,0.); \
	nproduce MoveTo(-0.4,-1.,-10.); \
	nproduce SetColor(1) SetWidth(w); \
    nproduce CurrentContour(1) TextureVCoeff(1.0/h); \
    nproduce CurrentTexture(day); \
    nproduce SB StartGC() F(h) EndGC() EB; \
    nproduce CurrentTexture(day+MAX_DAYS) RollR(90); \
    nproduce SB StartGC() F(h) EndGC() EB; \
    sprintf (str, "Day %d\n", day+1); \
	produce MoveTo(-1.5,8.,0.) Label(str); \
}

vgroup 0:


vgroup view001:

InternodeInterpretation(0)

LeafInterpretation(0)

ViualAidInterpretation(0)


vgroup view003:

InternodeInterpretation(1)

LeafInterpretation(1)

ViualAidInterpretation(2)


vgroup view006:

InternodeInterpretation(2)

LeafInterpretation(2)

ViualAidInterpretation(5)


vgroup view009:

InternodeInterpretation(3)

LeafInterpretation(3)

ViualAidInterpretation(8)


vgroup view012:

InternodeInterpretation(4)

LeafInterpretation(4)

ViualAidInterpretation(11)


vgroup view015:

InternodeInterpretation(5)

LeafInterpretation(5)

ViualAidInterpretation(14)


vgroup view018:

InternodeInterpretation(6)

LeafInterpretation(6)

ViualAidInterpretation(17)


vgroup view021:

InternodeInterpretation(7)

LeafInterpretation(7)

ViualAidInterpretation(20)


vgroup view024:

InternodeInterpretation(8)

LeafInterpretation(8)

ViualAidInterpretation(23)


vgroup view027:

InternodeInterpretation(9)

LeafInterpretation(9)

ViualAidInterpretation(26)

