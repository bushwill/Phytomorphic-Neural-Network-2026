#include <iostream>
#include <algorithm>
#include <cmath>

// Include all GLM core / GLSL features
#include "glm/glm.hpp" // vec2, vec3, mat4, radians

// Include all GLM extensions
#include "glm/ext.hpp" // perspective, translate, rotate

#include "glm/gtx/string_cast.hpp"

const float _MinimumNearClipping = 0.001f;
const float _DefaultScale = 1.1f;
const float _epsilon = 0.0001f;

int _WindowSize_w = 306;
int _WindowSize_h = 256;

//const float clip_Back = ;
//const float clip_Front = ;



float _sratio; // window size ratio width / height
glm::vec3 _viewCenter;

float _minWidth, _minHeight;
float _vratio; // view volume ratio minWidth / minHeight
float _HalfDepth;
float _upp; // units per pixel

float _scale;
float _fov;
float _ZShift;

glm::vec3 _viewDir, _viewPan;
glm::vec3 _viewUp, _viewLeft;


//view: 0 dir: 0 0 -1 up: 0 1 0 pan: 0 0 0 fov: 45 scale: 1 shift: 24.7201
//box: 0 -6.54391 5.74391 -1.1 9.18 -9.9089 9.9089

void ProjectionSetModifiers() { //const WindowParams::ViewModifiers &vm) {
  _scale = 1.;//vm.scale;
  _fov = 45.;//vm.fov;

  _viewPan = glm::vec3(0,0,0);//vm.viewPan;

  _viewDir = glm::vec3(0,0,-1);//vm.viewDir;
  //_viewDir.Normalize();
  _viewUp = glm::vec3(0,1,0);//vm.viewUp;
  // check if the angle between Dir and Up is almost zero
  //if (fabs(_viewDir * _viewUp) >= 1.f - Projection::_epsilon) {
  //  _viewUp.Set(0.f, -_viewDir.Z(), _viewDir.Y());
  //}
  _viewLeft = glm::normalize(glm::cross(_viewDir,_viewUp));//(_viewDir % _viewUp).Normalize();
  _viewUp = glm::normalize(glm::cross(_viewLeft,_viewDir));//(_viewLeft % _viewDir).Normalize();
  //_upRotationAxis = _viewUp;

  _ZShift = 24.7201;//vm.ZShift;
  // If vm.ZShift was zero, that means that we must recalculate it.
  //if (_ZShift <= 0)
  //  _ZShift = _minWidth * _scale / tanf(Deg2Rad(_fov * 0.5f)) + _HalfDepth;
  
  _sratio = float(_WindowSize_w) / float(_WindowSize_h);
}

inline float Deg2Rad(float deg) { return deg * M_PI / 180.0f; }

void ProjectionSetVolume() { //Volume v, const Clipping &clip) {
  const float MinDim = 0.1f;
  const float HalfDepthCoef = 0.525f;
  const float MinHalfDepth = 0.5f;

  //box: 0 -6.54391 5.74391 -1.1 9.18 -9.9089 9.9089
  //  _boundingBox = v;
  //Vector3d v_min(v.Min());
  //Vector3d v_max(v.Max());

  float _minX = -6.54391;
  float _maxX = 5.74391;
  float _minY = -1.1;
  float _maxY = 9.18;
  float _minZ = -9.9089;
  float _maxZ = 9.9089;
  glm::vec3 v_min = glm::vec3(_minX,_minY,_minZ);
  glm::vec3 v_max = glm::vec3(_maxX,_maxY,_maxZ);

  _viewCenter = glm::vec3(0.5f * (_maxX + _minX), 0.5f * (_maxY + _minY), 0.5f * (_maxZ + _minZ));
  //_viewCenter.X(v.CX());
  //_viewCenter.Y(v.CY());
  //_viewCenter.Z(v.CZ());

  //_minWidth = 0.5f * (v_max.X() - v_min.X());
  //_minHeight = 0.5f * (v_max.Y() - v_min.Y());
  _minWidth = 0.5f * (v_max.x - v_min.x);
  _minHeight = 0.5f * (v_max.y - v_min.y);
  if (0.0f == _minWidth)
    _minWidth = MinDim * _minHeight;
  if (0.0f == _minHeight)
    _minHeight = MinDim * _minWidth;
  if (0.0f == _minWidth)
    _minWidth = _minHeight = MinDim;
  _HalfDepth = HalfDepthCoef * glm::distance(v_min,v_max);//Distance(v_min, v_max);
  if (0.0f == _HalfDepth)
    _HalfDepth = MinHalfDepth;

//  if (clip.Specified()) {
//    float b = clip_Back;
//    float f = clip_Front;
//    _HalfDepth = fabsf((b - f) * 0.5f); // why not HalfDepthCoef?
//  }

  _ZShift = _minWidth * _scale / std::tanh(Deg2Rad(_fov * 0.5f)) + _HalfDepth;
  _vratio = _minWidth / _minHeight;
}


int main(int argc, char *argv[]) {

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " width height inputfile" << std::endl;
    std::cerr << "\twidth: the width of the lpfg window / output image" << std::endl;
    std::cerr << "\theight: the height of the lpfg window / output image" << std::endl;
    std::cerr << "\tinputfile: name of file with 3D positions of organs,\n\t\twith x,y,z coordinates given on each line" << std::endl; 
    return 1;
  }

  _WindowSize_w = std::atoi(argv[1]);
  _WindowSize_h = std::atoi(argv[2]);
  if (_WindowSize_w < 0 || _WindowSize_h < 0) {
      std::cerr << "Error: window width and height must be greater than zero\n";
      return 1;
  }

  // set projection parameters
  ProjectionSetModifiers();

  ProjectionSetVolume();

  // construct the projection matrix
  float nearClip = _ZShift - _HalfDepth, farClip = _ZShift + _HalfDepth;
  if (nearClip < _MinimumNearClipping)
    nearClip = _MinimumNearClipping;

  float xSize, ySize;
  if (_sratio > _vratio) // window wider than it is high
  {
    ySize = _minHeight * _scale;
    xSize = ySize * _sratio;
    //glOrtho(-xSize, xSize, -ySize, ySize, nearClip, farClip);
  } else // window is higher than it is wide
  {
    xSize = _minWidth * _scale;
    ySize = xSize / _sratio;
    //glOrtho(-xSize, xSize, -ySize, ySize, nearClip, farClip);
  }
  glm::mat4 Proj = glm::ortho(-xSize, xSize, -ySize, ySize, nearClip, farClip);

  // set the model-view matrix
  glm::vec3 _viewLookAt = glm::vec3(_viewCenter + _viewPan);
  glm::vec3 _viewPos = glm::vec3(_viewLookAt - _ZShift * _viewDir);
  //gluLookAt(_viewPos.X(), _viewPos.Y(), _viewPos.Z(), _viewLookAt.X(),
  //          _viewLookAt.Y(), _viewLookAt.Z(), _viewUp.X(), _viewUp.Y(),
  //          _viewUp.Z());
  glm::mat4 Modelview = glm::lookAt(_viewPos, _viewLookAt, _viewUp);

  glm::mat4 transform = glm::mat4(Proj * Modelview);

  //std::cerr << "Projection * View * Model transformation matrix:\n";
  //std::cerr << glm::to_string(transform) << std::endl;
  //std::cerr << std::endl;
  
  // parse the points, and output the screen coordinates
  FILE *infile = NULL;

  if ((infile = fopen(argv[3], "r")) == NULL) {
    std::cerr << "Error: cannot open file " << argv[3] << std::endl;
    return(1);
  }

  char buf[256];
  char type;
  int node, bufN;
  float data[3];
  while (fgets(buf, sizeof(buf), infile) != NULL) {
    bufN = sscanf (buf, "%c %d %f %f %f", &type, &node, &data[0], &data[1], &data[2]);

    if (bufN != 5) {
        std::cout << buf;
        continue;
    }
 
    glm::vec4 pt(data[0], data[1], data[2], 1.);
   
    // apply transformation to get homogeneous clip space
    // vertex coordinate to clip coordinates
    glm::vec4 res = glm::vec4(transform * pt);
    //std::cerr << "vertex to clip coordinates:\n";
    //std::cerr << glm::to_string(res) << std::endl;

    // clipping
    res.x = std::max(std::min(res.x,res.w),-res.w);
    res.y = std::max(std::min(res.y,res.w),-res.w);
    res.z = std::max(std::min(res.z,res.w),-res.w);
    res.w = std::max(0.0000001f,res.w);

    // perspective divide (normalized-device coordinates)
    res.x /= res.w;
    res.y /= res.w;
    res.z /= res.w;

    //std::cerr << "normalized-device coordinate:\n";
    //std::cerr << glm::to_string(res) << std::endl;

    // glViewport(): viewport transform to raster space
    res.x = (res.x + 1.f) * 0.5f * (float(_WindowSize_w) - 1.f);
    res.y = (1.f - (res.y + 1.f) * 0.5f) * (float(_WindowSize_h) - 1.f);
    //res.x = (res.x + 1.f) * _WindowSize_w * 0.5f + res.x;
    //res.y = (res.y + 1.f) * _WindowSize_h * 0.5f + res.y;

    //std::cerr << "raster space:\n";
    //std::cerr << glm::to_string(res) << std::endl;
    
    //std::cout << type << " " << node << " " << data[0] << " " << data[1] << "  " << data[2] << " : " << int(res.x) << " " << int(res.y) << std::endl;
    std::cout << type << " " << node << " " << int(res.x) << " " << int(res.y) << std::endl;
  }

  if (infile != NULL)
    fclose(infile);


  return 0;
}

