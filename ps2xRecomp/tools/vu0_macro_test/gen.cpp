#include "ps2recomp/code_generator.h"
#include "ps2recomp/r5900_decoder.h"
#include <cstdio>
#include <vector>
#include <string>
using namespace ps2recomp;
int main(){
  R5900Decoder dec; std::vector<Symbol> syms; std::vector<Section> secs; CodeGenerator cg(syms,secs);
  std::vector<uint32_t> ws; unsigned seed=12345; auto rnd=[&](){seed=seed*1103515245+12345; return (seed>>8);} ;
  auto reg=[&](){ return 1+rnd()%31; };
  // upper special1 ops (funct 0x00-0x2f) and special2 (idx 0x00-0x3b + 0x3c..)
  for(int f=0; f<0x30; f++) for(int k=0;k<3;k++){ uint32_t dest=1+rnd()%15; ws.push_back(0x4A000000u|(1u<<25)|(dest<<21)|(reg()<<16)|(reg()<<11)|(reg()<<6)|f);} 
  int s2[]={0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2a,0x2c,0x2d,0x2e,0x30,0x31,0x38,0x39,0x3a};
  for(int idx: s2) for(int k=0;k<3;k++){ uint32_t dest=1+rnd()%15; ws.push_back(0x4A000000u|(1u<<25)|(dest<<21)|(reg()<<16)|(reg()<<11)|((idx>>2)<<6)|0x3c|(idx&3)); }
  FILE*o=fopen("cases.inc","w");
  for(size_t i=0;i<ws.size();i++){
    Instruction in=dec.decodeInstruction(0x100000,ws[i]);
    std::string c=cg.translateInstruction(in);
    fprintf(o,"static void case_%zu(uint8_t*rdram,R5900Context*ctx,PS2Runtime*runtime){ (void)rdram;(void)runtime; %s }\n",i,c.c_str());
  }
  fprintf(o,"static const uint32_t kWords[]={"); for(auto w:ws) fprintf(o,"0x%08xu,",w); fprintf(o,"};\n");
  fprintf(o,"static void (*kCases[])(uint8_t*,R5900Context*,PS2Runtime*)={"); for(size_t i=0;i<ws.size();i++) fprintf(o,"case_%zu,",i); fprintf(o,"};\n");
  fclose(o); printf("%zu cases\n",ws.size());
}
