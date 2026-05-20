"""
Reannotate HOGraspNet with MediaPipe world_landmarks (local version).

- S01 already extracted at SOURCE_DATA -- used directly, no download.
- S02-S99: downloaded from Dropbox to TMP_EXTRACT, processed, deleted.
- Output CSV appended incrementally -- safe to interrupt and resume.
- No leftover files: TMP_EXTRACT is deleted after each subject.

Run from grasp-model/ with venv activated:
    python scripts/ingestion/reannotate_mediapipe_local.py
"""

import csv
import subprocess
import time
import zipfile
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
NFS_BASE    = Path('/media/yareeez/94649A33649A1856/HOGraspNet')
SOURCE_DATA = NFS_BASE / 'source_data'          # S01 already here
TMP_EXTRACT = NFS_BASE / 'tmp_extract'          # downloaded subjects land here
CSV_IN      = Path('/media/yareeez/94649A33649A1856/DATOS DE TESIS/hograspnet_mano.csv')
CSV_OUT     = NFS_BASE / 'hograspnet_mediapipe.csv'
LOG_FAILED  = NFS_BASE / 'hograspnet_mediapipe_failed.txt'

# ── Constants ─────────────────────────────────────────────────────────────────
JOINTS = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
]
XYZ_COLS      = [f'{j}_{ax}' for j in JOINTS for ax in ('x', 'y', 'z')]
WRIST_IDX     = 0
INDEX_MCP_IDX = 5
TARGET_DIST   = 0.1

META_COLS = ['subject_id', 'sequence_id', 'cam', 'frame_id', 'grasp_type', 'contact_sum']
OUT_COLS  = META_COLS + XYZ_COLS

SUBJECT_URLS = {
    "S01": None,  # already in SOURCE_DATA, no download needed
    "S02": "https://www.dropbox.com/scl/fi/526mamb8nogb6gkbpmysv/GraspNet_S2_1_Source_data.zip?rlkey=cvygh7qev4amh52l01rz9ad1r&dl=1",
    "S03": "https://www.dropbox.com/scl/fi/deh4ga22v83qx698kmc6a/GraspNet_S3_1_Source_data.zip?rlkey=67v6tx3muncheqgb2zr7x3hyp&dl=1",
    "S04": "https://www.dropbox.com/scl/fi/itcv902cn6ag6fhfw7z3a/GraspNet_S4_1_Source_data.zip?rlkey=ty8z9xlmkbivfumfcqyiz890y&dl=1",
    "S05": "https://www.dropbox.com/scl/fi/cu2qyafhyxvqquwdsksds/GraspNet_S5_1_Source_data.zip?rlkey=dlbyhkpxo7ptq47k9a3sme3zz&dl=1",
    "S06": "https://www.dropbox.com/scl/fi/cok57cd7olp88sa6iqagz/GraspNet_S6_1_Source_data.zip?rlkey=awh4tqu9ulezm8h5nmzrvo6ez&dl=1",
    "S07": "https://www.dropbox.com/scl/fi/xsdgoc65cinmekuq4emy4/GraspNet_S7_1_Source_data.zip?rlkey=5xvya2f4qz877t0eqwx11653l&dl=1",
    "S08": "https://www.dropbox.com/scl/fi/kzbdc6y9wbjqtwegxt6ub/GraspNet_S8_1_Source_data.zip?rlkey=ossp04yg1pip6fn4j1we5e06c&dl=1",
    "S09": "https://www.dropbox.com/scl/fi/imnr1ks8ivb3csmvq9189/GraspNet_S9_1_Source_data.zip?rlkey=1aq25mm7ofudvt72795x4na1s&dl=1",
    "S10": "https://www.dropbox.com/scl/fi/dv6zbs1lrd85d5ih0rqqk/GraspNet_S10_1_Source_data.zip?rlkey=c1hxbpp8mtaon9vorfdyknhn1&dl=1",
    "S11": "https://www.dropbox.com/scl/fi/eptfmx0eardrwjnyecd54/GraspNet_S11_1_Source_data.zip?rlkey=xik0pux9eyurb4yi0kt2t4oza&dl=1",
    "S12": "https://www.dropbox.com/scl/fi/qcjmq765tf2yeoaswze6m/GraspNet_S12_1_Source_data.zip?rlkey=zzufgp078law40qcyjlataeuz&dl=1",
    "S13": "https://www.dropbox.com/scl/fi/ko2uojypaabxkjbra5p0k/GraspNet_S13_1_Source_data.zip?rlkey=u71mu6pl8j6cnbpq2zr7x3hyp&dl=1",
    "S14": "https://www.dropbox.com/scl/fi/scvfht72w4v8il4y3wq46/GraspNet_S14_1_Source_data.zip?rlkey=tt5pg77t307tmhz6u601ccas4&dl=1",
    "S15": "https://www.dropbox.com/scl/fi/kqx3jf9bcttwz3k9xkp8z/GraspNet_S15_1_Source_data.zip?rlkey=zcwhqk25wqho4oocqdyxqpmxy&dl=1",
    "S16": "https://www.dropbox.com/scl/fi/15kc4z0cimgn8vlb1nimt/GraspNet_S16_1_Source_data.zip?rlkey=tm2gju9ypme1yqnnkiuwbckw0&dl=1",
    "S17": "https://www.dropbox.com/scl/fi/iy6q8mnwtchqsvht1zh9x/GraspNet_S17_1_Source_data.zip?rlkey=nvp0nvmmqb3e9dc73ggbql15j&dl=1",
    "S18": "https://www.dropbox.com/scl/fi/wpwln5fj3q0n8qwr58ucx/GraspNet_S18_1_Source_data.zip?rlkey=thg4j71bizs1i5fthz77kqx4c&dl=1",
    "S19": "https://www.dropbox.com/scl/fi/k04xlukf0t8v1jv8pb8j5/GraspNet_S19_1_Source_data.zip?rlkey=t6ao3gv0n1zoxo4r7195y2f8y&dl=1",
    "S20": "https://www.dropbox.com/scl/fi/6tnpc6zot73rtpq4bmceh/GraspNet_S20_1_Source_data.zip?rlkey=onj2xlwftiutu26pefvphq0lh&dl=1",
    "S21": "https://www.dropbox.com/scl/fi/owuihethwpdgpyhnjcdcg/GraspNet_S21_1_Source_data.zip?rlkey=pelrugt27u9zbyr38rmzy2jr1&dl=1",
    "S22": "https://www.dropbox.com/scl/fi/ra8tte29y0v4j87av9c81/GraspNet_S22_1_Source_data.zip?rlkey=pxn5mw66lq4agy76xg7ofaq06&dl=1",
    "S23": "https://www.dropbox.com/scl/fi/217j61ahnh3rvuqbp4r4t/GraspNet_S23_1_Source_data.zip?rlkey=4ef1trgi44mljj48c73zitl76&dl=1",
    "S24": "https://www.dropbox.com/scl/fi/rc3gtttmincxefgjmlrg4/GraspNet_S24_1_Source_data.zip?rlkey=yhclr5oc0sq3k7q75i7f1rjbn&dl=1",
    "S25": "https://www.dropbox.com/scl/fi/kuq3hn1gem8v8ofvoh7wd/GraspNet_S25_1_Source_data.zip?rlkey=e1efiadfm9kjga86sgpg2wb2p&dl=1",
    "S26": "https://www.dropbox.com/scl/fi/3he1zo3eddqoiugqve3s4/GraspNet_S26_1_Source_data.zip?rlkey=yue9qy9jh3aqy2z5n1dz3kgxo&dl=1",
    "S27": "https://www.dropbox.com/scl/fi/dasbdozeqjc98lxpkkcme/GraspNet_S27_1_Source_data.zip?rlkey=drz78ufosaaxm0kpm3z0zdfcp&dl=1",
    "S28": "https://www.dropbox.com/scl/fi/3czenftir31q1twzo1dwd/GraspNet_S28_1_Source_data.zip?rlkey=q10vx1ik9ty70l9g4mdq803zy&dl=1",
    "S29": "https://www.dropbox.com/scl/fi/txa2bqin3x8vr67mxy6cw/GraspNet_S29_1_Source_data.zip?rlkey=kwoj6tdn49auqic29jz3exe9c&dl=1",
    "S30": "https://www.dropbox.com/scl/fi/e2b7zwxxt7eacggkkh52b/GraspNet_S30_1_Source_data.zip?rlkey=dvsbeai8h8de7k62rh8wq036m&dl=1",
    "S31": "https://www.dropbox.com/scl/fi/tlgxafamzyvluecowzne9/GraspNet_S31_1_Source_data.zip?rlkey=pesq650mor4k8c5xez590rhfr&dl=1",
    "S32": "https://www.dropbox.com/scl/fi/isrpc304yz5hrnmw93h71/GraspNet_S32_1_Source_data.zip?rlkey=m7imhjjishjvvgmjgw7wpjnzx&dl=1",
    "S33": "https://www.dropbox.com/scl/fi/b1jslpzldd1uhmzwwz81z/GraspNet_S33_1_Source_data.zip?rlkey=xj8pmr2cazcojdyybbby4l6sb&dl=1",
    "S34": "https://www.dropbox.com/scl/fi/peampb49x03t6c801m4sa/GraspNet_S34_1_Source_data.zip?rlkey=zjsqxm2cvw1srtgsmg1m3r71p&dl=1",
    "S35": "https://www.dropbox.com/scl/fi/q0ig20fefcgi32al6ea4p/GraspNet_S35_1_Source_data.zip?rlkey=y9aohv8edfrzckxsf8wp4ei2p&dl=1",
    "S36": "https://www.dropbox.com/scl/fi/8gthpnsu0bqnx3pqyp5wo/GraspNet_S36_1_Source_data.zip?rlkey=z6enl62lfw2ie6iyp7rg6kmjw&dl=1",
    "S37": "https://www.dropbox.com/scl/fi/bbl952u6pqu3h87mqkvql/GraspNet_S37_1_Source_data.zip?rlkey=sktsco24vknwfd21gffngi5wd&dl=1",
    "S38": "https://www.dropbox.com/scl/fi/6omdgiyqqi76ct81uhph3/GraspNet_S38_1_Source_data.zip?rlkey=6jygxol362aha9q3gpjdbivtx&dl=1",
    "S39": "https://www.dropbox.com/scl/fi/navuh28us6gwsxmp5z57j/GraspNet_S39_1_Source_data.zip?rlkey=hw514wbgdgfetrkllbypzt2hg&dl=1",
    "S40": "https://www.dropbox.com/scl/fi/sud9v7bsbaxgvayfo2gdp/GraspNet_S40_1_Source_data.zip?rlkey=sp23gsu0vsxdhgzkwht7waszt&dl=1",
    "S41": "https://www.dropbox.com/scl/fi/46vwy7yc2b6oyt77svyyc/GraspNet_S41_1_Source_data.zip?rlkey=3op5zhn8yykna33qr25tucqdj&dl=1",
    "S42": "https://www.dropbox.com/scl/fi/1adxuj8vgnlxzgdkhp08p/GraspNet_S42_1_Source_data.zip?rlkey=noeiryu0h0oyzv6rz3ovgxeuy&dl=1",
    "S43": "https://www.dropbox.com/scl/fi/xcrvukabjwfquylrficwz/GraspNet_S43_1_Source_data.zip?rlkey=879640ebzyc3n94sm3cagusti&dl=1",
    "S44": "https://www.dropbox.com/scl/fi/82s27mlecnqzdvevuvpj5/GraspNet_S44_1_Source_data.zip?rlkey=rv270bfcwza1y7khmmhl8e9bb&dl=1",
    "S45": "https://www.dropbox.com/scl/fi/khjh4241ijs7895jdyw7t/GraspNet_S45_1_Source_data.zip?rlkey=gdesyupjcmaqcvhbfzitp7qre&dl=1",
    "S46": "https://www.dropbox.com/scl/fi/darp11j7fmchxk6sstf5z/GraspNet_S46_1_Source_data.zip?rlkey=rev6t96voq1yfgk1ynkxh3wso&dl=1",
    "S47": "https://www.dropbox.com/scl/fi/pzfn3k84h4r0pcj3yf61k/GraspNet_S47_1_Source_data.zip?rlkey=zsrpnw5chs44pfoo5p73cf57u&dl=1",
    "S48": "https://www.dropbox.com/scl/fi/ak21kbdnfwp3gxdu7tzzn/GraspNet_S48_1_Source_data.zip?rlkey=7lmdcdaao3infsnssqk0aik2n&dl=1",
    "S49": "https://www.dropbox.com/scl/fi/jpfsdmdu6jpsegjksf137/GraspNet_S49_1_Source_data.zip?rlkey=9fy1mcz7nbtjr2sc2zizh02j3&dl=1",
    "S50": "https://www.dropbox.com/scl/fi/jzgtzrut17f0e6k2veuy5/GraspNet_S50_1_Source_data.zip?rlkey=ae672s8pplhrdplotlxltaqei&dl=1",
    "S51": "https://www.dropbox.com/scl/fi/f559t78pfnlkkxaenklij/GraspNet_S51_1_Source_data.zip?rlkey=px2dykzmfbk77fjgzf0apfjcj&dl=1",
    "S52": "https://www.dropbox.com/scl/fi/201ih43c4afrr97r3lk5j/GraspNet_S52_1_Source_data.zip?rlkey=k8xqc70hidf2bzd08mc9gsdgv&dl=1",
    "S53": "https://www.dropbox.com/scl/fi/w69hhz0t75y6gka0epa71/GraspNet_S53_1_Source_data.zip?rlkey=yed6ur8xbls6bn98oyklrao68&dl=1",
    "S54": "https://www.dropbox.com/scl/fi/s1614t4g1fgn54luz28lj/GraspNet_S54_1_Source_data.zip?rlkey=7iduxe1lyt6oo6j8w8ossb6tr&dl=1",
    "S55": "https://www.dropbox.com/scl/fi/50bpkmtd1l5d9xjrsvukw/GraspNet_S55_1_Source_data.zip?rlkey=3z1zh0sfx8mqm51gietsywrxa&dl=1",
    "S56": "https://www.dropbox.com/scl/fi/a5i9yah9fkcanwmz0220m/GraspNet_S56_1_Source_data.zip?rlkey=6ve6mm45eu8wj43n4hbcmwrml&dl=1",
    "S57": "https://www.dropbox.com/scl/fi/kaop4lu76at47k5ym0g0y/GraspNet_S57_1_Source_data.zip?rlkey=k5aij089pdk4j0zd5sn474gm5&dl=1",
    "S58": "https://www.dropbox.com/scl/fi/vtx675m5yxw9oznudhlpn/GraspNet_S58_1_Source_data.zip?rlkey=67nng8rixooxki1yrbf41q93g&dl=1",
    "S59": "https://www.dropbox.com/scl/fi/rmbvwyz9345lofk41blr7/GraspNet_S59_1_Source_data.zip?rlkey=2ekadxzhr4w3smq2jnwfot4eu&dl=1",
    "S60": "https://www.dropbox.com/scl/fi/tdc2l3ae0nz3tckuwf2ha/GraspNet_S60_1_Source_data.zip?rlkey=xw0dhfoh6d5l2r6i53bl2ixd7&dl=1",
    "S61": "https://www.dropbox.com/scl/fi/1ln8bph76hewxtmrmwpt4/GraspNet_S61_1_Source_data.zip?rlkey=j17dsr31l0c6ka13clqot3y42&dl=1",
    "S62": "https://www.dropbox.com/scl/fi/63vztus5o69kcamc6f8h5/GraspNet_S62_1_Source_data.zip?rlkey=8n0lz7imqmmjbz2agxsf26vvt&dl=1",
    "S63": "https://www.dropbox.com/scl/fi/pco9iflyqdpwcbgr6ubb9/GraspNet_S63_1_Source_data.zip?rlkey=u413qurxlhqbaak0zko7e2mes&dl=1",
    "S64": "https://www.dropbox.com/scl/fi/ocdzg8ptvdnq5ic4xosiv/GraspNet_S64_1_Source_data.zip?rlkey=g1mkwz9zoly2yyh4fsttt0wpx&dl=1",
    "S65": "https://www.dropbox.com/scl/fi/lp0ixut9t8w55r3q52ude/GraspNet_S65_1_Source_data.zip?rlkey=n7ohvcdfnfeaddxlaimgbk3a7&dl=1",
    "S66": "https://www.dropbox.com/scl/fi/von3cks1t35s9zl9kqih0/GraspNet_S66_1_Source_data.zip?rlkey=8h8b2kzi6ogq81pe1svze9ah7&dl=1",
    "S67": "https://www.dropbox.com/scl/fi/4fnkqowk74d52zk54nvt0/GraspNet_S67_1_Source_data.zip?rlkey=q2qege9mybubsezlhatb4oe9b&dl=1",
    "S68": "https://www.dropbox.com/scl/fi/drcjmapmufcvaroh753wv/GraspNet_S68_1_Source_data.zip?rlkey=8jylv4dgbx0obu8m713j7wjka&dl=1",
    "S69": "https://www.dropbox.com/scl/fi/0htontuvv5ilb8ddn95lh/GraspNet_S69_1_Source_data.zip?rlkey=z84wj115g5zc06a0kkaou0qbs&dl=1",
    "S70": "https://www.dropbox.com/scl/fi/16auws6d2c7lalrc850yr/GraspNet_S70_1_Source_data.zip?rlkey=jh7dcjnilh6j2ghjjb9ix2at0&dl=1",
    "S71": "https://www.dropbox.com/scl/fi/z9prvnf41ixn99zpaipzb/GraspNet_S71_1_Source_data.zip?rlkey=subaa2ze3ztwo9jf1t5d88g9x&dl=1",
    "S72": "https://www.dropbox.com/scl/fi/q4xggvxpumx6o8ohb5hpy/GraspNet_S72_1_Source_data.zip?rlkey=3x5tnjy4paxfme37ozhojg7v7&dl=1",
    "S73": "https://www.dropbox.com/scl/fi/kx9avgb968wimyb9xadhp/GraspNet_S73_1_Source_data.zip?rlkey=0d5p0a1x7wqm55hqr81rehdfw&dl=1",
    "S74": "https://www.dropbox.com/scl/fi/gd9r718f8glwaomm9clgc/GraspNet_S74_1_Source_data.zip?rlkey=d9qgnghdfsdy7al30eb7jotil&dl=1",
    "S75": "https://www.dropbox.com/scl/fi/nbjufvf4wobrtgdsrp9ip/GraspNet_S75_1_Source_data.zip?rlkey=tevb8b7bodbh2ujxaskoyhqzo&dl=1",
    "S76": "https://www.dropbox.com/scl/fi/loqyt2ppt6mfdqpilwdw7/GraspNet_S76_1_Source_data.zip?rlkey=jjnv5hk12xvm402baki7niynf&dl=1",
    "S77": "https://www.dropbox.com/scl/fi/d34ci4ovyozl8sz05an65/GraspNet_S77_1_Source_data.zip?rlkey=z6xxu183udiwdusqk9tpvn29r&dl=1",
    "S78": "https://www.dropbox.com/scl/fi/x0scxwo0zg4tqh1ab9wju/GraspNet_S78_1_Source_data.zip?rlkey=b8ngsoupj2o1epwxyuyemvka0&dl=1",
    "S79": "https://www.dropbox.com/scl/fi/sw51urjomg1moldzmkuuz/GraspNet_S79_1_Source_data.zip?rlkey=ksuzdcc4mrv02lj90d1ho1bk7&dl=1",
    "S80": "https://www.dropbox.com/scl/fi/f7rxpwzigryicpx1li2zq/GraspNet_S80_1_Source_data.zip?rlkey=0ndms22vnjkgy2y0cjicytqef&dl=1",
    "S81": "https://www.dropbox.com/scl/fi/hi7u86v8r4cwth8wrnlda/GraspNet_S81_1_Source_data.zip?rlkey=icz1u5ulqo7n9ab7qogc89xzp&dl=1",
    "S82": "https://www.dropbox.com/scl/fi/n2473lp9uyjef366mnl8w/GraspNet_S82_1_Source_data.zip?rlkey=sycc3d3qqcypxps9ryqw4y4rp&dl=1",
    "S83": "https://www.dropbox.com/scl/fi/p78w8maf9oo1i0fjh483r/GraspNet_S83_1_Source_data.zip?rlkey=4dxvormejql176b74ut7n6hda&dl=1",
    "S84": "https://www.dropbox.com/scl/fi/d7itae6vy60761jj7c01r/GraspNet_S84_1_Source_data.zip?rlkey=pxer8phlpstgx2kvf5mfvo7io&dl=1",
    "S85": "https://www.dropbox.com/scl/fi/g2drsl9mw5nh4jur1ttiw/GraspNet_S85_1_Source_data.zip?rlkey=vwrn7eu1apbrcebhes50d7gt5&dl=1",
    "S86": "https://www.dropbox.com/scl/fi/0eoh1b7j2y9lrjzcx7ruj/GraspNet_S86_1_Source_data.zip?rlkey=lxp1u63fsp7iy6pspswh3ddfx&dl=1",
    "S87": "https://www.dropbox.com/scl/fi/zf9xpnq5pansdeyju2q9h/GraspNet_S87_1_Source_data.zip?rlkey=kzu4cixwy7fswzkvqqn8xinhv&dl=1",
    "S88": "https://www.dropbox.com/scl/fi/allwpijqk9wiek53ltdrr/GraspNet_S88_1_Source_data.zip?rlkey=fmikd1thdc10nzlgw5g2m2dpb&dl=1",
    "S89": "https://www.dropbox.com/scl/fi/xfms0i0rf47nz8dk5gqjo/GraspNet_S89_1_Source_data.zip?rlkey=n17ene1t1wg1oibqklzfa9k5h&dl=1",
    "S90": "https://www.dropbox.com/scl/fi/3r8z7nv3zfyt1musikrl4/GraspNet_S90_1_Source_data.zip?rlkey=ofh4sa0cgfr1kounhwgfonfx4&dl=1",
    "S91": "https://www.dropbox.com/scl/fi/6h6ahn9xdgdo14nlh7wod/GraspNet_S91_1_Source_data.zip?rlkey=y0ozvr3tskube2hem01pscelc&dl=1",
    "S92": "https://www.dropbox.com/scl/fi/n19m53z373gupeyz5bf9t/GraspNet_S92_1_Source_data.zip?rlkey=t8sn9qog5czh3s8ek7gsr9o48&dl=1",
    "S93": "https://www.dropbox.com/scl/fi/5bf64tpfq51xgd5ox3plj/GraspNet_S93_1_Source_data.zip?rlkey=bwmhart71h97jini3azlgocjo&dl=1",
    "S94": "https://www.dropbox.com/scl/fi/eigc3c8qbo3x3twawzmdb/GraspNet_S94_1_Source_data.zip?rlkey=6px73zxa45x1u85sqlfj81ff5&dl=1",
    "S95": "https://www.dropbox.com/scl/fi/ma6bmj8fcaihzp9xwlnh7/GraspNet_S95_1_Source_data.zip?rlkey=dw5n0cd93zoqk8537q35qk4z1&dl=1",
    "S96": "https://www.dropbox.com/scl/fi/hg2riwhvykbbmcspm6i9k/GraspNet_S96_1_Source_data.zip?rlkey=1qai2hyxmu1m21tjt191tcsk9&dl=1",
    "S97": "https://www.dropbox.com/scl/fi/yirunais37m4g9wpqo5ux/GraspNet_S97_1_Source_data.zip?rlkey=tyaxm1i8wzm01vgkc5by0uznw&dl=1",
    "S98": "https://www.dropbox.com/scl/fi/0epxalfc3mjcx1o8vu91s/GraspNet_S98_1_Source_data.zip?rlkey=wtjsyax1ow2tvbk8j2lh0m206&dl=1",
    "S99": "https://www.dropbox.com/scl/fi/lvy394b559220boreruxd/GraspNet_S99_1_Source_data.zip?rlkey=2aao3o2obv5jey3cjb576pg3z&dl=1",
}


def normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts.copy()
    pts -= pts[WRIST_IDX]
    d = np.linalg.norm(pts[INDEX_MCP_IDX])
    if d > 1e-6:
        pts *= TARGET_DIST / d
    return pts


def img_path(extract_dir: Path, sequence_id: str, cam: str, frame_id: int) -> Path:
    parts = sequence_id.split('/')
    return extract_dir / parts[0] / parts[1] / 'rgb' / cam / f'{cam}_{frame_id}.jpg'


def download_and_extract(url: str, subject: str) -> Path:
    TMP_EXTRACT.mkdir(parents=True, exist_ok=True)
    zip_path = TMP_EXTRACT / f'{subject}.zip'
    for attempt in range(1, 4):
        print(f'  Downloading {subject} (attempt {attempt})...')
        subprocess.run(['wget', '-q', '--show-progress', '-O', str(zip_path), url], check=True)
        size_mb = zip_path.stat().st_size / 1024 / 1024
        if size_mb < 10:
            print(f'  Download too small ({size_mb:.1f} MB) -- Dropbox error, retrying...')
            zip_path.unlink(missing_ok=True)
            time.sleep(5)
            continue
        try:
            print(f'  Extracting {subject}...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(TMP_EXTRACT)
            zip_path.unlink()
            return TMP_EXTRACT
        except zipfile.BadZipFile:
            print(f'  Bad zip file -- retrying...')
            zip_path.unlink(missing_ok=True)
            time.sleep(5)
    print(f'  Skipping {subject} -- download failed after 3 attempts (Dropbox error).')
    return None


def cleanup():
    if TMP_EXTRACT.exists():
        shutil.rmtree(TMP_EXTRACT)
        print(f'  Cleaned {TMP_EXTRACT}')


def main():
    print(f'Loading input CSV...')
    _avail = pd.read_csv(CSV_IN, nrows=0).columns.tolist()
    _use   = [c for c in META_COLS if c in _avail]
    df_all = pd.read_csv(CSV_IN, usecols=_use)
    total_frames = len(df_all)
    print(f'Total frames: {total_frames:,}')

    # Resume: load successfully processed frames
    done_keys = set()
    if CSV_OUT.exists():
        done_df = pd.read_csv(CSV_OUT, usecols=['sequence_id', 'cam', 'frame_id'])
        for _, r in done_df.iterrows():
            done_keys.add((r['sequence_id'], r['cam'], int(r['frame_id'])))
        print(f'Resuming -- already done: {len(done_keys):,}')

    # Also skip frames that previously failed (no_hand, missing, etc.)
    # They are almost always unrecoverable -- no need to re-download zips for them
    if LOG_FAILED.exists():
        import re
        with open(LOG_FAILED) as f:
            for line in f:
                # format: reason|/path/to/cam_frameid.jpg
                m = re.search(r'/([^/]+)/([^/]+)\.jpg$', line.strip())
                if not m:
                    continue
                # extract cam and frame_id from filename: sub3_83 -> cam=sub3, frame_id=83
                filename = m.group(2)  # e.g. sub3_83
                parts = filename.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                cam = parts[0]
                try:
                    frame_id = int(parts[1])
                except ValueError:
                    continue
                # extract sequence_id from path: .../seq_base/trial_X/rgb/cam/...
                path_parts = line.strip().split('|')[-1].split('/')
                try:
                    rgb_idx = path_parts.index('rgb')
                    trial   = path_parts[rgb_idx - 1]
                    seq_base = path_parts[rgb_idx - 2]
                    done_keys.add((f'{seq_base}/{trial}', cam, frame_id))
                except (ValueError, IndexError):
                    continue
        print(f'After skipping failed frames: {len(done_keys):,} total skipped')

    already_done = len(done_keys)
    write_header = not CSV_OUT.exists()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

    failed_log = open(LOG_FAILED, 'a')
    global_ok = global_fail = 0
    t_global = time.time()

    with open(CSV_OUT, 'a', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=OUT_COLS)
        if write_header:
            writer.writeheader()

        for subject, url in SUBJECT_URLS.items():
            subj_id = int(subject[1:])
            df_subj = df_all[df_all['subject_id'] == subj_id].reset_index(drop=True)
            if len(df_subj) == 0:
                print(f'[{subject}] No frames, skipping.')
                continue

            pending = [
                r for _, r in df_subj.iterrows()
                if (r['sequence_id'], r['cam'], int(r['frame_id'])) not in done_keys
            ]
            if not pending:
                print(f'[{subject}] Already complete ({len(df_subj):,} frames), skipping.')
                continue

            print(f'\n[{subject}] {len(pending):,} frames to process')

            # S01 already on disk
            if url is None:
                extract_dir = SOURCE_DATA
            else:
                extract_dir = download_and_extract(url, subject)
                if extract_dir is None:
                    continue

            n_ok = n_fail = 0
            t_start = time.time()

            for row in pending:
                seq_id   = row['sequence_id']
                cam      = row['cam']
                frame_id = int(row['frame_id'])

                path = img_path(extract_dir, seq_id, cam, frame_id)
                if not path.exists():
                    failed_log.write(f'missing|{path}\n'); failed_log.flush()
                    n_fail += 1; continue

                frame = cv2.imread(str(path))
                if frame is None:
                    failed_log.write(f'read_error|{path}\n'); failed_log.flush()
                    n_fail += 1; continue

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if not result.multi_hand_world_landmarks:
                    failed_log.write(f'no_hand|{path}\n'); failed_log.flush()
                    n_fail += 1; continue

                wlms = result.multi_hand_world_landmarks[0]
                kp   = np.array([[lm.x, lm.y, lm.z] for lm in wlms.landmark], dtype=np.float32)

                if result.multi_handedness:
                    label    = result.multi_handedness[0].classification[0].label
                    is_right = (label == 'Left')  # selfie flip
                else:
                    is_right = True

                kp_n = normalize(kp)
                if not is_right:
                    kp_n[:, 0] *= -1.0

                out_row = {c: row[c] for c in _use if c in OUT_COLS}
                for j, joint in enumerate(JOINTS):
                    out_row[f'{joint}_x'] = float(kp_n[j, 0])
                    out_row[f'{joint}_y'] = float(kp_n[j, 1])
                    out_row[f'{joint}_z'] = float(kp_n[j, 2])

                writer.writerow(out_row)
                f_out.flush()
                done_keys.add((seq_id, cam, frame_id))
                n_ok += 1

                done = n_ok + n_fail
                if done % 500 == 0:
                    elapsed     = time.time() - t_start
                    rate        = done / elapsed if elapsed > 0 else 0
                    global_done = already_done + global_ok + global_fail + done
                    remaining   = (total_frames - global_done) / rate if rate > 0 else 0
                    print(f'  [{global_done:,}/{total_frames:,}] ok={n_ok} fail={n_fail} | {rate:.1f} frames/s | ETA {remaining/60:.1f} min')

            elapsed = time.time() - t_start
            rate    = (n_ok + n_fail) / elapsed if elapsed > 0 else 0
            print(f'  [{subject}] Done. ok={n_ok} fail={n_fail} | {rate:.1f} frames/s | {elapsed/60:.1f} min')
            global_ok   += n_ok
            global_fail += n_fail

            if url is not None:
                cleanup()

    total_elapsed = time.time() - t_global
    hands.close()
    failed_log.close()
    print(f'\nAll done. ok={global_ok:,} fail={global_fail:,} | total {total_elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
