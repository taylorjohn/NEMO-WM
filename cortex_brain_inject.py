import math, time, struct
from collections import deque
import numpy as np

class NeuroState:
    def __init__(self):
        self.da=0.5; self.sht=1.0; self.ne=0.2; self.ach=0.5
        self.ecb=0.0; self.cortisol=0.0; self.ei=1.0
        self.regime="EXPLOIT"; self.z_entry=2.92; self.pos_scale=1.0

class DopamineSystem:
    def __init__(self):
        self.s=NeuroState(); self._z_hist=deque(maxlen=20)
        self._adverse=0.0; self._z_pred=0.0; self._n=0
    def update(self,z,rho,is_toxic,rtt_ms,qty):
        s=self.s; self._n+=1
        rpe=abs(z-self._z_pred)/(abs(self._z_pred)+0.1)
        s.da=0.95*s.da+0.05*min(1.0,rpe); self._z_pred=z
        stab=0.8
        s.sht=0.90*s.sht+0.10*stab
        s.ne=min(1.0,max(0.0,rho/10.0))
        s.ach=0.85*s.ach+0.15*(min(1.0,rpe)+(1.0-stab))/2.0
        s.ecb=0.85*s.ecb+0.15*s.da*min(1.0,qty/10.0)
        da_eff=s.da*(1.0-s.ecb*0.4)
        adv=float(is_toxic or rtt_ms>120.0) if self._n>200 else 0.0
        self._adverse=self._adverse*0.9+adv
        s.cortisol=min(2.0,0.97*s.cortisol+0.03*max(0.0,self._adverse/10.0-0.2))
        s.ei=float(np.clip(da_eff/(1.0-s.sht+0.1),0.5,2.0))
        s.regime="WAIT" if s.sht<0.05 else "EXPLORE" if da_eff>0.65 else "EXPLOIT"
        s.z_entry=2.92*(1.0+s.cortisol*0.5)
        s.pos_scale=max(0.05,1.0-s.cortisol*0.7)
        return s

class BrainInjector:
    def __init__(self,engine):
        self.e=engine; self.brain=DopamineSystem()
        self._tick=0; self._ns=NeuroState()
        print("Brain injector active")
        print("5HT=RTT-based | cortisol=spread+RTT only | WAIT@sht<0.05")
    def run(self):
        e=self.e
        while True:
            try:
                self._tick+=1
                rho=e.get_rho()
                e.rho_history.append(rho)
                if len(e.rho_history)>e.history_limit: e.rho_history.pop(0)
                try:
                    import cortex_math
                    z=cortex_math.calculate_z_score(e.rho_history,rho)
                except Exception:
                    arr=np.array(e.rho_history)
                    z=float((rho-arr.mean())/(arr.std()+1e-6)) if len(arr)>2 else 0.0
                e.check_market_hours()
                if e.market_open:
                    e.update_account_state(); e.manage_fills()
                    ns=self.brain.update(z,rho,e.is_toxic_state,e.last_rtt,e.cached_qty)
                    self._ns=ns
                    if e.dynamic_z_threshold!=-999.0: e.dynamic_z_threshold=ns.z_entry
                    if not e.active_order_id and ns.regime!="WAIT":
                        saved=e.mirror_wallet; e.mirror_wallet=saved*ns.pos_scale
                        e.live_execute(z); e.mirror_wallet=saved
                    elif ns.regime=="WAIT" and self._tick%200==0:
                        print(f"WAIT rtt={e.last_rtt:.1f}ms 5HT={ns.sht:.3f}")
                if self._tick%100==0:
                    ns=self._ns
                    print(f"[t={self._tick}] z={z:.3f} DA={ns.da:.3f} 5HT={ns.sht:.3f} rtt={e.last_rtt:.1f}ms CORT={ns.cortisol:.3f} regime={ns.regime} z_entry={ns.z_entry:.3f} scale={ns.pos_scale:.2f} wallet=${e.mirror_wallet:.2f}")
                e.update_engine_state()
                try:
                    ns=self._ns
                    pkt=struct.pack("<dffffi",time.time(),float(e.mirror_wallet),float(z),float(e.last_rtt),float(e.dynamic_z_threshold),int(e.engine_state))
                    e.sock.sendto(pkt,("192.168.1.150",5005))
                except Exception: pass
                time.sleep(0.05)
            except KeyboardInterrupt:
                print("Brain shutdown"); break
            except Exception as ex:
                print(f"RECOVERY: {ex}"); time.sleep(2)
