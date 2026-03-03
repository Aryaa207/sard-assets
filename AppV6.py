"""
AppV5 - IMU Dashboard
Single script: serves HTML + WebSocket on port 5000
Run: python3 AppV5.py
Open: http://localhost:5000
"""
import asyncio, math, time, json, threading, board, busio
from http.server import HTTPServer, BaseHTTPRequestHandler
import adafruit_lsm9ds1

# ── HTML PAGE (embedded) ────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARGUS-IV IMU</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@700&display=swap');
:root{--bg:#0a0d12;--panel:#111820;--border:#253040;--amber:#ffb000;--white:#dce4eb;--dim:#5a6470;--red:#e03030;--blue:#3ca0ff;--green:#3cd278;--dark:#181f2a;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--white);font-family:'Share Tech Mono',monospace;height:100vh;display:flex;flex-direction:column;overflow:hidden;}
/* TOP */
.top{background:var(--panel);border-bottom:1px solid var(--border);padding:0 18px;height:42px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}
.title{font-family:'Rajdhani',sans-serif;font-size:16px;font-weight:700;letter-spacing:3px;display:flex;align-items:center;gap:10px;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--red);transition:background .3s;}
.clock{font-size:11px;color:var(--amber);}
/* MAIN GRID */
.main{display:grid;grid-template-columns:200px 1fr 280px;flex:1;gap:1px;background:var(--border);overflow:hidden;}
.panel{background:var(--panel);display:flex;flex-direction:column;overflow:hidden;}
.ph{padding:6px 12px;border-bottom:1px solid var(--border);font-size:10px;color:var(--amber);letter-spacing:2px;font-family:'Rajdhani',sans-serif;font-weight:700;flex-shrink:0;}
/* LEFT DATA */
.dp{padding:10px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;}
.st{font-size:9px;color:var(--amber);letter-spacing:2px;border-bottom:1px solid var(--dark);padding-bottom:2px;margin-top:4px;}
.dr{display:flex;justify-content:space-between;align-items:center;}
.dl{font-size:10px;color:var(--dim);}
.dv{font-size:12px;font-weight:bold;font-family:'Rajdhani',sans-serif;}
.raw{color:var(--red)}.kal{color:var(--green)}.acc{color:var(--blue)}
.bt{background:var(--dark);height:4px;border-radius:2px;position:relative;margin:1px 0 4px;}
.bm{position:absolute;left:50%;top:-2px;width:1px;height:8px;background:var(--border);}
.bf{position:absolute;height:4px;border-radius:2px;top:0;will-change:left,width;}
/* VIEWER */
.vp{position:relative;}
#c{width:100%;height:100%;display:block;}
/* PLOTS */
.rp{overflow-y:auto;}
.pw{padding:8px 10px;border-bottom:1px solid var(--dark);}
.pt{font-size:9px;color:var(--amber);letter-spacing:1px;margin-bottom:4px;}
canvas.p{width:100%!important;height:68px!important;}
/* BOTTOM */
.bot{background:var(--panel);border-top:1px solid var(--border);padding:0 18px;height:32px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;font-size:10px;color:var(--dim);}
</style>
</head>
<body>
<div class="top">
  <div class="title"><div class="dot" id="dot"></div>ARGUS-IV &nbsp;//&nbsp; IMU TELEMETRY</div>
  <div class="clock" id="clk"></div>
</div>
<div class="main">
  <!-- LEFT -->
  <div class="panel"><div class="ph">SENSOR DATA</div><div class="dp">
    <div class="st">ROLL</div>
    <div class="dr"><span class="dl">RAW</span><span class="dv raw" id="rr">--</span></div>
    <div class="dr"><span class="dl">KALMAN</span><span class="dv kal" id="rk">--</span></div>
    <div class="bt"><div class="bm"></div><div class="bf" id="brr" style="background:var(--red);opacity:.5"></div><div class="bf" id="brk" style="background:var(--green);height:2px;top:1px"></div></div>
    <div class="st">PITCH</div>
    <div class="dr"><span class="dl">RAW</span><span class="dv raw" id="pr">--</span></div>
    <div class="dr"><span class="dl">KALMAN</span><span class="dv kal" id="pk">--</span></div>
    <div class="bt"><div class="bm"></div><div class="bf" id="bpr" style="background:var(--red);opacity:.5"></div><div class="bf" id="bpk" style="background:var(--green);height:2px;top:1px"></div></div>
    <div class="st">YAW</div>
    <div class="dr"><span class="dl">GYRO</span><span class="dv kal" id="yk">--</span></div>
    <div class="st">GYRO (dps)</div>
    <div class="dr"><span class="dl">X</span><span class="dv" id="gx">--</span></div>
    <div class="dr"><span class="dl">Y</span><span class="dv" id="gy">--</span></div>
    <div class="dr"><span class="dl">Z</span><span class="dv" id="gz">--</span></div>
    <div class="st">ACCEL (g)</div>
    <div class="dr"><span class="dl">X</span><span class="dv acc" id="ax">--</span></div>
    <div class="dr"><span class="dl">Y</span><span class="dv acc" id="ay">--</span></div>
    <div class="dr"><span class="dl">Z</span><span class="dv acc" id="az">--</span></div>
  </div></div>
  <!-- CENTRE -->
  <div class="panel vp"><div class="ph">3D ORIENTATION // KALMAN FILTERED</div><canvas id="c"></canvas></div>
  <!-- RIGHT -->
  <div class="panel rp"><div class="ph">LIVE PLOTS</div>
    <div class="pw"><div class="pt">ROLL &mdash; RED=raw &nbsp; GRN=kalman</div><canvas class="p" id="cr"></canvas></div>
    <div class="pw"><div class="pt">PITCH &mdash; RED=raw &nbsp; GRN=kalman</div><canvas class="p" id="cp"></canvas></div>
    <div class="pw"><div class="pt">YAW &mdash; gyro integrated</div><canvas class="p" id="cy"></canvas></div>
    <div class="pw"><div class="pt">ACCEL XYZ (g)</div><canvas class="p" id="ca"></canvas></div>
  </div>
</div>
<div class="bot">
  <span>LSM9DS1 9-DOF // KALMAN FILTER // 50Hz</span>
  <span id="wst" style="color:var(--red)">CONNECTING...</span>
  <span>TOP SECRET // SCI</span>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script>
// CLOCK
setInterval(()=>{document.getElementById('clk').textContent=new Date().toUTCString().split(' ')[4]+' UTC';},500);

// THREE.JS
const cv=document.getElementById('c');
const R=new THREE.WebGLRenderer({canvas:cv,antialias:false});
R.setPixelRatio(1); R.setClearColor(0x0a0d12);
const S=new THREE.Scene();
const CAM=new THREE.PerspectiveCamera(50,1,0.1,100);
CAM.position.set(0,2.5,5); CAM.lookAt(0,0,0);
S.add(new THREE.GridHelper(10,20,0x1e2632,0x1e2632));
S.add(new THREE.AmbientLight(0xffffff,0.8));
const dl=new THREE.DirectionalLight(0xffb000,1.8); dl.position.set(5,10,5); S.add(dl);
S.add(new THREE.AxesHelper(1.5));

// BUILD DRONE
const D=new THREE.Group();
const BM=new THREE.MeshPhongMaterial({color:0x1a2535,shininess:80});
const AM=new THREE.MeshPhongMaterial({color:0x0f151e});
const PM=new THREE.MeshPhongMaterial({color:0x3ca0ff,transparent:true,opacity:0.75});
const MM=new THREE.MeshPhongMaterial({color:0xffb000,shininess:120});

// body
const body=new THREE.Mesh(new THREE.CylinderGeometry(0.38,0.32,0.12,6),BM);
D.add(body);
const topPlate=new THREE.Mesh(new THREE.CylinderGeometry(0.28,0.28,0.05,6),new THREE.MeshPhongMaterial({color:0x253040}));
topPlate.position.set(0,0.08,0); D.add(topPlate);

// 4 arms
[[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([sx,sz])=>{
  const ex=sx*0.85, ez=sz*0.85;
  const arm=new THREE.Mesh(new THREE.CylinderGeometry(0.035,0.035,1.1,6),AM);
  arm.rotation.z=Math.PI/2; arm.rotation.y=Math.atan2(sx,sz);
  arm.position.set(ex*0.5,0,ez*0.5); D.add(arm);
  const mot=new THREE.Mesh(new THREE.CylinderGeometry(0.1,0.1,0.09,10),MM);
  mot.position.set(ex,0.04,ez); D.add(mot);
  const pd=new THREE.Mesh(new THREE.CylinderGeometry(0.3,0.3,0.012,16),PM);
  pd.position.set(ex,0.11,ez); D.add(pd);
  const b1=new THREE.Mesh(new THREE.BoxGeometry(0.55,0.01,0.065),PM);
  b1.position.set(ex,0.12,ez); D.add(b1);
  const b2=new THREE.Mesh(new THREE.BoxGeometry(0.065,0.01,0.55),PM);
  b2.position.set(ex,0.12,ez); D.add(b2);
  const leg=new THREE.Mesh(new THREE.CylinderGeometry(0.018,0.018,0.28,6),AM);
  leg.position.set(ex*0.75,-0.18,ez*0.75); D.add(leg);
});
// skids
[-1,1].forEach(s=>{
  const sk=new THREE.Mesh(new THREE.CylinderGeometry(0.018,0.018,1.2,6),AM);
  sk.rotation.z=Math.PI/2; sk.rotation.y=Math.PI/4;
  sk.position.set(s*0.25,-0.32,0); D.add(sk);
});
// camera
const cg=new THREE.Mesh(new THREE.BoxGeometry(0.14,0.12,0.16),new THREE.MeshPhongMaterial({color:0x111111}));
cg.position.set(0,-0.1,0.32); D.add(cg);
const lens=new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,0.07,8),new THREE.MeshPhongMaterial({color:0x000000}));
lens.rotation.x=Math.PI/2; lens.position.set(0,-0.1,0.44); D.add(lens);
// gps dome
const gps=new THREE.Mesh(new THREE.SphereGeometry(0.07,8,6),new THREE.MeshPhongMaterial({color:0xffffff}));
gps.position.set(0,0.15,-0.12); D.add(gps);
S.add(D);

let tR=0,tP=0,tY=0;
function loop(){
  requestAnimationFrame(loop);
  const w=cv.clientWidth,h=cv.clientHeight;
  if(R.domElement.width!==w||R.domElement.height!==h){R.setSize(w,h,false);CAM.aspect=w/h;CAM.updateProjectionMatrix();}
  D.rotation.x=THREE.MathUtils.degToRad(tP);
  D.rotation.z=THREE.MathUtils.degToRad(-tR);
  D.rotation.y=THREE.MathUtils.degToRad(tY);
  R.render(S,CAM);
}
loop();

// CHARTS
const N=80;
function mkC(id,cs){
  return new Chart(document.getElementById(id),{
    type:'line',
    data:{labels:Array(N).fill(''),datasets:cs.map(c=>({data:Array(N).fill(null),borderColor:c,borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}))},
    options:{animation:false,responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{display:false},y:{grid:{color:'#181f2a'},ticks:{color:'#5a6470',font:{size:9}}}}}
  });
}
const CR=mkC('cr',['#e03030','#3cd278']);
const CP=mkC('cp',['#e03030','#3cd278']);
const CY=mkC('cy',['#ffb000']);
const CA=mkC('ca',['#3ca0ff','#3cd278','#ffb000']);
function push(ch,...vs){ch.data.datasets.forEach((d,i)=>{d.data.push(vs[i]);if(d.data.length>N)d.data.shift();});ch.update('none');}

// BARS
function bar(id,a){
  const e=document.getElementById(id);
  const p=Math.max(0,Math.min(100,(a+90)/180*100)),m=50;
  if(p>=m){e.style.left=m+'%';e.style.width=(p-m)+'%';}
  else{e.style.left=p+'%';e.style.width=(m-p)+'%';}
}

// WEBSOCKET
let ct=0;
function connect(){
  const ws=new WebSocket('ws://'+location.host.replace(':5000','')+':8765');
  ws.onopen=()=>{
    document.getElementById('wst').textContent='CONNECTED';
    document.getElementById('wst').style.color='#3cd278';
    document.getElementById('dot').style.background='#3cd278';
  };
  ws.onclose=()=>{
    document.getElementById('wst').textContent='RECONNECTING...';
    document.getElementById('wst').style.color='#e03030';
    document.getElementById('dot').style.background='#e03030';
    setTimeout(connect,1000);
  };
  ws.onmessage=(e)=>{
    const d=JSON.parse(e.data);
    tR=d.rk; tP=d.pk; tY=d.yk;
    document.getElementById('rr').textContent=d.rr.toFixed(1)+'°';
    document.getElementById('rk').textContent=d.rk.toFixed(1)+'°';
    document.getElementById('pr').textContent=d.pr.toFixed(1)+'°';
    document.getElementById('pk').textContent=d.pk.toFixed(1)+'°';
    document.getElementById('yk').textContent=d.yk.toFixed(1)+'°';
    document.getElementById('gx').textContent=d.gx.toFixed(2);
    document.getElementById('gy').textContent=d.gy.toFixed(2);
    document.getElementById('gz').textContent=d.gz.toFixed(2);
    document.getElementById('ax').textContent=d.ax.toFixed(3);
    document.getElementById('ay').textContent=d.ay.toFixed(3);
    document.getElementById('az').textContent=d.az.toFixed(3);
    bar('brr',d.rr); bar('brk',d.rk); bar('bpr',d.pr); bar('bpk',d.pk);
    if(++ct%4===0){push(CR,d.rr,d.rk);push(CP,d.pr,d.pk);push(CY,d.yk);push(CA,d.ax,d.ay,d.az);}
  };
}
connect();
</script>
</body>
</html>"""

# ── KALMAN ──────────────────────────────────────────────────
class KF:
    def __init__(self):
        self.angle=0.0; self.bias=0.0
        self.P=[[0.0,0.0],[0.0,0.0]]
        self.Q_a=0.001; self.Q_b=0.003; self.R=0.03
    def reset(self,a): self.angle=a; self.bias=0.0
    def update(self,a,r,dt):
        self.angle+=dt*(r-self.bias)
        self.P[0][0]+=dt*(dt*self.P[1][1]-self.P[0][1]-self.P[1][0]+self.Q_a)
        self.P[0][1]-=dt*self.P[1][1]; self.P[1][0]-=dt*self.P[1][1]
        self.P[1][1]+=self.Q_b*dt
        S=self.P[0][0]+self.R; K=[self.P[0][0]/S,self.P[1][0]/S]
        y=a-self.angle; self.angle+=K[0]*y; self.bias+=K[1]*y
        p00,p01=self.P[0][0],self.P[0][1]
        self.P[0][0]-=K[0]*p00; self.P[0][1]-=K[0]*p01
        self.P[1][0]-=K[1]*p00; self.P[1][1]-=K[1]*p01
        return self.angle

# ── SENSOR ──────────────────────────────────────────────────
print("Connecting to LSM9DS1...")
i2c = busio.I2C(board.SCL, board.SDA)
imu = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)

print("Calibrating gyro (keep still 2s)...")
bx=by=bz=0.0
for _ in range(100):
    gx,gy,gz=imu.gyro
    bx+=gx; by+=gy; bz+=gz
    time.sleep(0.02)
bx/=100; by/=100; bz/=100
print(f"Bias X={bx:.3f} Y={by:.3f} Z={bz:.3f}")

kfr=KF(); kfp=KF()
ax0,ay0,az0=imu.acceleration
kfr.reset(math.degrees(math.atan2(ay0,az0)))
kfp.reset(math.degrees(math.atan2(-ax0,math.sqrt(ay0**2+az0**2))))

# ── SHARED STATE ────────────────────────────────────────────
import websockets
CLIENTS = set()
yaw = 0.0
prev_t = time.perf_counter()

# ── HTTP SERVER (serves the HTML page) ──────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type','text/html')
        self.end_headers()
        self.wfile.write(HTML.encode())
    def log_message(self, *args): pass  # silence logs

def run_http():
    server = HTTPServer(('0.0.0.0', 5000), Handler)
    server.serve_forever()

threading.Thread(target=run_http, daemon=True).start()
print("Dashboard at http://localhost:5000")

# ── WEBSOCKET SERVER ─────────────────────────────────────────
async def sensor_loop():
    global yaw, prev_t, CLIENTS
    while True:
        now = time.perf_counter()
        dt  = max(now-prev_t, 0.001)
        prev_t = now

        ax,ay,az = imu.acceleration
        gx,gy,gz = imu.gyro
        gx-=bx; gy-=by; gz-=bz

        rr = math.degrees(math.atan2(ay,az))
        pr = math.degrees(math.atan2(-ax,math.sqrt(ay**2+az**2)))
        yaw += gz*dt
        rk = kfr.update(rr,gx,dt)
        pk = kfp.update(pr,gy,dt)

        if CLIENTS:
            msg = json.dumps({
                "rr":round(rr,2),"pr":round(pr,2),
                "rk":round(rk,2),"pk":round(pk,2),"yk":round(yaw,2),
                "gx":round(gx,3),"gy":round(gy,3),"gz":round(gz,3),
                "ax":round(ax,3),"ay":round(ay,3),"az":round(az,3),
            })
            dead=set()
            for ws in CLIENTS.copy():
                try: await ws.send(msg)
                except: dead.add(ws)
            CLIENTS -= dead

        await asyncio.sleep(0.02)  # 50Hz

async def ws_handler(ws):
    global CLIENTS
    CLIENTS.add(ws)
    print(f"Browser connected ({len(CLIENTS)} clients)")
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)

async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        await sensor_loop()

print("WebSocket at ws://0.0.0.0:8765")
print("Open http://localhost:5000 in your browser\n")
asyncio.run(main())
