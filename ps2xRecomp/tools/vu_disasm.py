import struct,sys
BC='xyzw'
def dest(i):
    d=(i>>21)&0xf; return ''.join(c for c,b in zip('xyzw',(8,4,2,1)) if d&b)
def upper(i):
    op=i&0x3f; ft=(i>>16)&31; fs=(i>>11)&31; fd=(i>>6)&31; bc=BC[i&3]; ds=dest(i)
    flags=''.join(f for f,b in (('[I]',31),('[E]',30),('[M]',29),('[D]',28),('[T]',27)) if (i>>b)&1)
    if op<0x3c:
        t1={0:'ADD',1:'SUB',2:'MADD',3:'MSUB',4:'MAX',5:'MINI',6:'MUL'}
        if op<0x1c: return '%s%s.%s vf%d,vf%d,vf%d%s'%(t1[op>>2],bc,ds,fd,fs,ft,flags)
        t={0x1c:'MULq',0x1d:'MAXi',0x1e:'MULi',0x1f:'MINIi',0x20:'ADDq',0x21:'MADDq',0x22:'ADDi',0x23:'MADDi',0x24:'SUBq',0x25:'MSUBq',0x26:'SUBi',0x27:'MSUBi',0x28:'ADD',0x29:'MADD',0x2a:'MUL',0x2b:'MAX',0x2c:'SUB',0x2d:'MSUB',0x2e:'OPMSUB',0x2f:'MINI'}
        n=t.get(op,'?u%x'%op)
        if op>=0x28: return '%s.%s vf%d,vf%d,vf%d%s'%(n,ds,fd,fs,ft,flags)
        return '%s.%s vf%d,vf%d%s'%(n,ds,fd,fs,flags)
    idx=(i&3)|((i>>4)&0x7c)
    t2={0x1c:'MULAq',0x1d:'ABS',0x1e:'MULAi',0x1f:'CLIP',0x20:'ADDAq',0x21:'MADDAq',0x22:'ADDAi',0x23:'MADDAi',0x24:'SUBAq',0x25:'MSUBAq',0x26:'SUBAi',0x27:'MSUBAi',0x28:'ADDA',0x29:'MADDA',0x2a:'MULA',0x2c:'SUBA',0x2d:'MSUBA',0x2e:'OPMULA',0x2f:'NOP'}
    if idx<0x10: return '%sA%s.%s ACC,vf%d,vf%d%s'%({0:'ADD',1:'SUB',2:'MADD',3:'MSUB'}[idx>>2],BC[idx&3],ds,fs,ft,flags)
    if idx<0x14: return 'ITOF%s.%s vf%d,vf%d%s'%((0,4,12,15)[idx&3],ds,ft,fs,flags)
    if idx<0x18: return 'FTOI%s.%s vf%d,vf%d%s'%((0,4,12,15)[idx&3],ds,ft,fs,flags)
    if idx<0x1c: return 'MULA%s.%s ACC,vf%d,vf%d%s'%(BC[idx&3],ds,fs,ft,flags)
    n=t2.get(idx,'?u2_%x'%idx)
    if n=='NOP': return 'NOP'+flags
    return '%s.%s vf%d,vf%d%s'%(n,ds,fs,ft,flags)
def sx(v,b): return v-(1<<b) if v&(1<<(b-1)) else v
def lower(i,pc):
    top=i>>25; ft=(i>>16)&31; fs=(i>>11)&31; fd=(i>>6)&31; ds=dest(i); imm11=sx(i&0x7ff,11)
    if top==0x40:
        op=i&0x3f
        if op<0x3c:
            t={0x30:'IADD',0x31:'ISUB',0x32:'IADDI',0x34:'IAND',0x35:'IOR'}
            n=t.get(op,'?l%x'%op)
            if n=='IADDI': return 'IADDI vi%d,vi%d,%d'%(ft,fs,sx((i>>6)&31,5))
            return '%s vi%d,vi%d,vi%d'%(n,fd,fs,ft)
        idx=(i&3)|((i>>4)&0x7c)
        t={0x30:'MOVE',0x31:'MR32',0x34:'LQI',0x35:'SQI',0x36:'LQD',0x37:'SQD',0x38:'DIV',0x39:'SQRT',0x3a:'RSQRT',0x3b:'WAITQ',0x3c:'MTIR',0x3d:'MFIR',0x3e:'ILWR',0x3f:'ISWR',0x40:'RNEXT',0x41:'RGET',0x42:'RINIT',0x43:'RXOR',0x64:'MFP',0x68:'XTOP',0x69:'XITOP',0x6c:'XGKICK',0x70:'ESADD',0x71:'ERSADD',0x72:'ELENG',0x73:'ERLENG',0x74:'EATANxy',0x75:'EATANxz',0x76:'ESUM',0x78:'ESQRT',0x79:'ERSQRT',0x7a:'ERCPR',0x7b:'WAITP',0x7c:'ESIN',0x7d:'EATAN',0x7e:'EEXP'}
        n=t.get(idx,'?l2_%x'%idx)
        fsf='xyzw'[(i>>21)&3]; ftf='xyzw'[(i>>23)&3]
        if n in('MOVE','MR32'): return '%s.%s vf%d,vf%d'%(n,ds,ft,fs)
        if n in('LQI','LQD'): return '%s.%s vf%d,(vi%d%s)'%(n,ds,ft,fs,'++' if n=='LQI' else '--')
        if n in('SQI','SQD'): return '%s.%s vf%d,(vi%d%s)'%(n,ds,fs,ft,'++' if n=='SQI' else '--')
        if n in('DIV','RSQRT'): return '%s Q,vf%d%s,vf%d%s'%(n,fs,fsf,ft,ftf)
        if n=='SQRT': return 'SQRT Q,vf%d%s'%(ft,ftf)
        if n=='MTIR': return 'MTIR vi%d,vf%d%s'%(ft,fs,fsf)
        if n=='MFIR': return 'MFIR.%s vf%d,vi%d'%(ds,ft,fs)
        if n in('ILWR','ISWR'): return '%s.%s vi%d,(vi%d)'%(n,ds,ft,fs)
        if n in('XTOP','XITOP'): return '%s vi%d'%(n,ft)
        if n=='XGKICK': return 'XGKICK vi%d'%fs
        if n=='MFP': return 'MFP.%s vf%d,P'%(ds,ft)
        if n[0]=='E': return '%s P,vf%d%s'%(n,fs,fsf)
        return n+' ft=%d fs=%d'%(ft,fs)
    t={0:'LQ',1:'SQ',4:'ILW',5:'ISW',8:'IADDIU',9:'ISUBIU',0x10:'FCEQ',0x11:'FCSET',0x12:'FCAND',0x13:'FCOR',0x14:'FSEQ',0x15:'FSSET',0x16:'FSAND',0x17:'FSOR',0x18:'FMEQ',0x1a:'FMAND',0x1b:'FMOR',0x1c:'FCGET',0x20:'B',0x21:'BAL',0x24:'JR',0x25:'JALR',0x28:'IBEQ',0x29:'IBNE',0x2c:'IBLTZ',0x2d:'IBGTZ',0x2e:'IBLEZ',0x2f:'IBGEZ'}
    n=t.get(top,'?L%x'%top)
    tgt=lambda: '0x%x'%((pc+8+imm11*8)&0x3fff)
    if n=='LQ': return 'LQ.%s vf%d,%d(vi%d)'%(ds,ft,imm11,fs)
    if n=='SQ': return 'SQ.%s vf%d,%d(vi%d)'%(ds,fs,imm11,ft)
    if n in('ILW','ISW'): return '%s.%s vi%d,%d(vi%d)'%(n,ds,ft,imm11,fs)
    if n in('IADDIU','ISUBIU'): return '%s vi%d,vi%d,%d'%(n,ft,fs,((i>>10)&0x7800)|(i&0x7ff))
    if n in('FCSET','FCEQ','FCAND','FCOR'): return '%s 0x%x'%(n,i&0xffffff)
    if n in('FMAND','FMEQ','FMOR','FSAND','FSEQ','FSOR','FSSET'): return '%s vi%d,vi%d 0x%x'%(n,ft,fs,i&0xfff)
    if n=='FCGET': return 'FCGET vi%d'%ft
    if n in('B','BAL'): return '%s vi%d,%s'%(n,ft,tgt())
    if n in('JR','JALR'): return '%s vi%d,vi%d'%(n,ft,fs)
    if n.startswith('IB'): return '%s vi%d,vi%d,%s'%(n,ft,fs,tgt())
    return n+' 0x%x'%i
def dis(code,start,count):
    pc=start; out=[]
    for k in range(count):
        lo,up=struct.unpack_from('<II',code,pc)
        u=upper(up)
        l = ('LOI %f'%struct.unpack('<f',struct.pack('<I',lo))[0]) if (up>>31)&1 else lower(lo,pc)
        out.append('%04x: %-40s %s'%(pc,u,l))
        pc+=8
    return out
if __name__=='__main__':
    d=open(sys.argv[1],'rb').read()
    code=d[4+12+512+64+16+16:4+12+512+64+16+16+16384]
    st=int(sys.argv[2],16); n=int(sys.argv[3])
    print('\n'.join(dis(code,st,n)))
