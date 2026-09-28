#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include "cases.inc"
struct V{float v[4];};
static V ld(__m128 m){V r; _mm_storeu_ps(r.v,m); return r;}
static int32_t ftoi(float f){ if(!(std::fabs(f)<2147483520.f)) return f<0?INT32_MIN:INT32_MAX; return (int32_t)f; }
int main(){
  std::mt19937 rng(7); std::uniform_real_distribution<float> U(-100.f,100.f);
  int bad=0,n=sizeof(kWords)/4;
  for(int i=0;i<n;i++){
    static R5900Context c; std::memset(&c,0,sizeof(c));
    V vf[32],acc; float q=U(rng), I=U(rng);
    for(int r=0;r<32;r++) for(int k=0;k<4;k++) vf[r].v[k]= r==0 ? (k==3?1.f:0.f) : U(rng);
    for(int k=0;k<4;k++) acc.v[k]=U(rng);
    uint32_t w=kWords[i]; bool isItof=false;
    // make ITOF inputs integers
    int idx=((w&3)|((w>>4)&0x7c)); bool s2=(w&0x3f)>=0x3c;
    if(s2 && idx>=0x10 && idx<0x14){ int fs=(w>>11)&31; for(int k=0;k<4;k++){ int32_t iv=(int32_t)(U(rng)*1000); std::memcpy(&vf[fs].v[k],&iv,4);} isItof=true; }
    for(int r=0;r<32;r++) c.vu0_vf[r]=_mm_loadu_ps(vf[r].v);
    c.vu0_acc=_mm_loadu_ps(acc.v); c.vu0_q=q; c.vu0_i=I;
    kCases[i](nullptr,&c,nullptr);
    // reference
    V E[32]; for(int r=0;r<32;r++) E[r]=vf[r]; V EA=acc; float EQ=q;
    int d=(w>>21)&15, ft=(w>>16)&31, fs=(w>>11)&31, fd=(w>>6)&31, f=w&0x3f;
    auto on=[&](int k){ return (d>>(3-k))&1; };
    const char* name="?";
    auto apply=[&](V& dst, auto fn, bool xyzOnly=false){ for(int k=0;k<4;k++) if(on(k) && !(xyzOnly&&k==3)) dst.v[k]=fn(k); };
    const V S=vf[fs], T=vf[ft];
    if(!s2){
      int bc=f&3;
      switch(f>>2){
        case 0: name="ADDbc"; apply(E[fd],[&](int k){return S.v[k]+T.v[bc];}); break;
        case 1: name="SUBbc"; apply(E[fd],[&](int k){return S.v[k]-T.v[bc];}); break;
        case 2: name="MADDbc"; apply(E[fd],[&](int k){return acc.v[k]+S.v[k]*T.v[bc];}); break;
        case 3: name="MSUBbc"; apply(E[fd],[&](int k){return acc.v[k]-S.v[k]*T.v[bc];}); break;
        case 4: name="MAXbc"; apply(E[fd],[&](int k){return std::max(S.v[k],T.v[bc]);}); break;
        case 5: name="MINIbc"; apply(E[fd],[&](int k){return std::min(S.v[k],T.v[bc]);}); break;
        case 6: name="MULbc"; apply(E[fd],[&](int k){return S.v[k]*T.v[bc];}); break;
        default:
        switch(f){
          case 0x1c: name="MULq"; apply(E[fd],[&](int k){return S.v[k]*q;}); break;
          case 0x1d: name="MAXi"; apply(E[fd],[&](int k){return std::max(S.v[k],I);}); break;
          case 0x1e: name="MULi"; apply(E[fd],[&](int k){return S.v[k]*I;}); break;
          case 0x1f: name="MINIi"; apply(E[fd],[&](int k){return std::min(S.v[k],I);}); break;
          case 0x20: name="ADDq"; apply(E[fd],[&](int k){return S.v[k]+q;}); break;
          case 0x21: name="MADDq"; apply(E[fd],[&](int k){return acc.v[k]+S.v[k]*q;}); break;
          case 0x22: name="ADDi"; apply(E[fd],[&](int k){return S.v[k]+I;}); break;
          case 0x23: name="MADDi"; apply(E[fd],[&](int k){return acc.v[k]+S.v[k]*I;}); break;
          case 0x24: name="SUBq"; apply(E[fd],[&](int k){return S.v[k]-q;}); break;
          case 0x25: name="MSUBq"; apply(E[fd],[&](int k){return acc.v[k]-S.v[k]*q;}); break;
          case 0x26: name="SUBi"; apply(E[fd],[&](int k){return S.v[k]-I;}); break;
          case 0x27: name="MSUBi"; apply(E[fd],[&](int k){return acc.v[k]-S.v[k]*I;}); break;
          case 0x28: name="ADD"; apply(E[fd],[&](int k){return S.v[k]+T.v[k];}); break;
          case 0x29: name="MADD"; apply(E[fd],[&](int k){return acc.v[k]+S.v[k]*T.v[k];}); break;
          case 0x2a: name="MUL"; apply(E[fd],[&](int k){return S.v[k]*T.v[k];}); break;
          case 0x2b: name="MAX"; apply(E[fd],[&](int k){return std::max(S.v[k],T.v[k]);}); break;
          case 0x2c: name="SUB"; apply(E[fd],[&](int k){return S.v[k]-T.v[k];}); break;
          case 0x2d: name="MSUB"; apply(E[fd],[&](int k){return acc.v[k]-S.v[k]*T.v[k];}); break;
          case 0x2e: name="OPMSUB"; { const int a[3]={1,2,0},b[3]={2,0,1}; apply(E[fd],[&](int k){return acc.v[k]-S.v[a[k]]*T.v[b[k]];},true);} break;
          case 0x2f: name="MINI"; apply(E[fd],[&](int k){return std::min(S.v[k],T.v[k]);}); break;
          default: continue;
        }
      }
    } else {
      int bc=idx&3; int fsf=d&3, ftf=d>>2;
      if(idx<0x10){ static const char* nm[4]={"ADDAbc","SUBAbc","MADDAbc","MSUBAbc"}; name=nm[idx>>2];
        apply(EA,[&](int k){ switch(idx>>2){case 0:return S.v[k]+T.v[bc];case 1:return S.v[k]-T.v[bc];case 2:return acc.v[k]+S.v[k]*T.v[bc];default:return acc.v[k]-S.v[k]*T.v[bc];}}); }
      else if(idx<0x14){ name="ITOF"; static const int sh[4]={0,4,12,15}; apply(E[ft],[&](int k){int32_t iv; std::memcpy(&iv,&S.v[k],4); return (float)iv/(float)(1<<sh[idx&3]);}); }
      else if(idx<0x18){ name="FTOI"; static const int sh[4]={0,4,12,15}; apply(E[ft],[&](int k){int32_t iv=ftoi(S.v[k]*(float)(1<<sh[idx&3])); float r; std::memcpy(&r,&iv,4); return r;}); }
      else if(idx<0x1c){ name="MULAbc"; apply(EA,[&](int k){return S.v[k]*T.v[bc];}); }
      else switch(idx){
        case 0x1c: name="MULAq"; apply(EA,[&](int k){return S.v[k]*q;}); break;
        case 0x1d: name="ABS"; apply(E[ft],[&](int k){return std::fabs(S.v[k]);}); break;
        case 0x1e: name="MULAi"; apply(EA,[&](int k){return S.v[k]*I;}); break;
        case 0x20: name="ADDAq"; apply(EA,[&](int k){return S.v[k]+q;}); break;
        case 0x21: name="MADDAq"; apply(EA,[&](int k){return acc.v[k]+S.v[k]*q;}); break;
        case 0x22: name="ADDAi"; apply(EA,[&](int k){return S.v[k]+I;}); break;
        case 0x23: name="MADDAi"; apply(EA,[&](int k){return acc.v[k]+S.v[k]*I;}); break;
        case 0x24: name="SUBAq"; apply(EA,[&](int k){return S.v[k]-q;}); break;
        case 0x25: name="MSUBAq"; apply(EA,[&](int k){return acc.v[k]-S.v[k]*q;}); break;
        case 0x26: name="SUBAi"; apply(EA,[&](int k){return S.v[k]-I;}); break;
        case 0x27: name="MSUBAi"; apply(EA,[&](int k){return acc.v[k]-S.v[k]*I;}); break;
        case 0x28: name="ADDA"; apply(EA,[&](int k){return S.v[k]+T.v[k];}); break;
        case 0x29: name="MADDA"; apply(EA,[&](int k){return acc.v[k]+S.v[k]*T.v[k];}); break;
        case 0x2a: name="MULA"; apply(EA,[&](int k){return S.v[k]*T.v[k];}); break;
        case 0x2c: name="SUBA"; apply(EA,[&](int k){return S.v[k]-T.v[k];}); break;
        case 0x2d: name="MSUBA"; apply(EA,[&](int k){return acc.v[k]-S.v[k]*T.v[k];}); break;
        case 0x2e: name="OPMULA"; { const int a[3]={1,2,0},b[3]={2,0,1}; apply(EA,[&](int k){return S.v[a[k]]*T.v[b[k]];},true);} break;
        case 0x30: name="MOVE"; apply(E[ft],[&](int k){return S.v[k];}); break;
        case 0x31: name="MR32"; apply(E[ft],[&](int k){return S.v[(k+1)&3];}); break;
        case 0x38: name="DIV"; EQ=S.v[fsf]/T.v[ftf]; break;
        case 0x39: name="SQRT"; EQ=std::sqrt(std::fabs(T.v[ftf])); break;
        case 0x3a: name="RSQRT"; EQ=S.v[fsf]/std::sqrt(std::fabs(T.v[ftf])); break;
        default: continue;
      }
    }
    E[0]=vf[0];
    auto close=[](float a,float b){ return std::memcmp(&a,&b,4)==0 || std::fabs(a-b)<=1e-5f*std::max(1.f,std::fabs(b)); };
    bool ok=true; char msg[256]="";
    for(int r=0;r<32&&ok;r++){ V g=ld(c.vu0_vf[r]); for(int k=0;k<4;k++) if(!close(g.v[k],E[r].v[k])){ok=false; snprintf(msg,sizeof msg,"vf%d[%d] got %g want %g",r,k,g.v[k],E[r].v[k]); break;} }
    V ga=ld(c.vu0_acc); for(int k=0;k<4&&ok;k++) if(!close(ga.v[k],EA.v[k])){ok=false; snprintf(msg,sizeof msg,"acc[%d] got %g want %g",k,ga.v[k],EA.v[k]);}
    if(ok && !close(c.vu0_q,EQ)){ok=false; snprintf(msg,sizeof msg,"Q got %g want %g",c.vu0_q,EQ);}
    if(!ok){ bad++; printf("FAIL %-8s w=%08x d=%x: %s\n",name,w,d,msg); }
  }
  printf("%d/%d failed\n",bad,n);
}
