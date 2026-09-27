import struct,sys,collections
def parse(fn,verbose=40):
    d=open(fn,'rb').read()
    regs=struct.unpack_from('<23I',d,4); size=struct.unpack_from('<I',d,4+92)[0]
    s=d[100:100+size]
    names={0:'NOP',1:'STCYCL',2:'OFFSET',3:'BASE',4:'ITOP',5:'STMOD',6:'MSKPATH3',7:'MARK',0x10:'FLUSHE',0x11:'FLUSH',0x13:'FLUSHA',0x14:'MSCAL',0x15:'MSCALF',0x17:'MSCNT',0x20:'STMASK',0x30:'STROW',0x31:'STCOL',0x4a:'MPG',0x50:'DIRECT',0x51:'DIRECTHL'}
    print(fn,'size',size,'regs cycle=%x mode=%x mask=%x base=%x ofst=%x tops=%x row=%s'%(regs[4],regs[5],regs[7],regs[10],regs[11],regs[12],[hex(x) for x in regs[15:19]]))
    pos=0;n=0;cnt=collections.Counter()
    while pos+4<=len(s):
        cmd=struct.unpack_from('<I',s,pos)[0]; pos+=4
        op=(cmd>>24)&0x7f; imm=cmd&0xffff; num=(cmd>>16)&0xff
        if (op&0x60)==0x60:
            vn=(op>>2)&3; vl=op&3; m=(op>>4)&1
            bits={0:32,1:16,2:8,3:16 if vn!=3 else 4}[vl]
            bpv=2 if (vl==3 and vn==3) else (vn+1)*bits//8
            cnt['UNPACK V%d-%d%s'%(vn+1,{0:32,1:16,2:8,3:5}[vl],' m' if m else '')]+=1
            w=num or 256
            ln=((w*bpv)+3)&~3
            if n<verbose: print('  %05x UNPACK V%d-%d m=%d num=%d addr=%x flg=%d usn=%d'%(pos-4,vn+1,{0:32,1:16,2:8,3:5}[vl],m,num,imm&0x3ff,(imm>>15)&1,(imm>>14)&1))
            pos+=ln
        else:
            nm=names.get(op,'?%x'%op); cnt[nm]+=1
            if n<verbose: print('  %05x %s imm=%x num=%d'%(pos-4,nm,imm,num))
            if op==0x20: pos+=4
            elif op in (0x30,0x31):
                if n<verbose: print('     data',[hex(x) for x in struct.unpack_from('<4I',s,pos)])
                pos+=16
            elif op==0x4a: pos+=(num or 256)*8
            elif op in (0x50,0x51): pos+=(imm or 65536)*16
        n+=1
    print(' counts',dict(cnt))
for fn in sys.argv[1:]: parse(fn)
