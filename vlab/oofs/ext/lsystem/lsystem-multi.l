/* ********************************************************************
 * Copyright (c) 2021 Biological Modeling and Visualization Laboratory,
 * University of Calgary. All rights reserved.
 * ********************************************************************/

/* Descriptive model of maize */
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

/* Module definitions */

module A(float,float); // age, phytomer count
module I(float,float);
module L(float,float);
module C(float,float); // leaf blade
module VisualAid();

/* Start of L-system */

derivation length: STEPS;

Axiom: SB Down(4.) RollL(90.) A(val(PLASTOCHRON),0) EB VisualAid;

production:

A(age,n) : {
    age += DAGE*DT;
	produce A(age,n);
}

L(age,n) : {
    age += DAGE*DT;
	produce L(age,n);
}

I(age,n) : {
    age += DAGE*DT;
    produce I(age,n);
}

decomposition:
maximum depth: 100;

A(age,n) : {
	if (age >= val(PLASTOCHRON) && n < val(MAX_PHYTOMERS)) {
        age = age - val(PLASTOCHRON); // decrement apex age
		produce I(age,n) // create new internode
			SB  // start branch
                // branching angle
                Down(val(BrAngle) * br_target_angle(n))
                L(age,n) // create leaf at position n
            EB // end branch
			RollL(180.0) // distichous phyllotaxis
			A(age,n+1); // recreate apex
	}
}

interpretation:
maximum depth: 100;

I(age,n): { 
    if (age <= 0.) 
        produce; 
    // calculate internode length and width
    float len = val(IntLen) * internode_target_len(n) * internode_length(age); 
    float wid = std::powf(len,val(ExpIntRad)) * val(IntWid) * internode_width(n); 
    int colour = (int(n) % 3) + 2; 
    // draw a cylinder
    produce SetColor(colour) SetWidth(wid) F(len); 
} 

L(age,n): { 
    if (age <= 0.) 
        produce;
    // calculate leaf length
	float len = val(LeafLen) * leaf_target_len(n) * leaf_length(age);
    // set the initial leaf width
    nproduce SetWidth(std::powf(len,val(ExpLeafWid)) * val(LeafWid) * leaf_width(0)); 
    // start drawing the generalized cylinder
	nproduce  
        Down(br_angle(age)) // NOTE: fixed from published version; br_angle(age) must be here.
		StartGC();
        float x = 0.f;
        for (x = 0.; x < 1.; x += DX) { 
            // determine current leaf cross-section
            nproduce BlendedContour(0,1,std::pow(x,0.2)); // NOTE: changed interpolation to x^{0.2}
            // generate leaf segment
            nproduce F(len*DX); 
            // set width and orientation of next segment
		    nproduce SetWidth(std::powf(len,val(ExpLeafWid)) * val(LeafWid) * leaf_width(x)) 
                    Down(leaf_bend(x)*len*DX) 
                    RollL(leaf_twist(x)*len*DX); 
        } 
    nproduce F(len*(1.-x)); // generate last segment
    produce EndGC(); // end the generalized cylinder
} 

VisualAid() : { 
	const float SF = 0.01; 
	float w = 1227.0 * SF; 
	float h = 1028.0 * SF; 
    nproduce SetElasticity(0,0.); 
	nproduce MoveTo(-0.4,-1.,0.); 
	nproduce SetColor(1) SetWidth(w); 
    nproduce CurrentContour(1) TextureVCoeff(1.0/h); 
    nproduce CurrentTexture(0); 
    nproduce SB Down(90) f(5.) Up(90) StartGC() F(h) EndGC() EB; 
    nproduce CurrentTexture(1); 
    nproduce Down(90) f(-0.4) Up(90) Left(90) f(-0.4) Right(90) RollR(90); 
    nproduce SB StartGC() F(h) EndGC() EB; 
    produce;
}

