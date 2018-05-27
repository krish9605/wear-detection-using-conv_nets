import math
import time
import random
import sys, getopt
# import http.client as httpc

# from six.moves import http_client as httpc
import httplib as httpc
import requests as rq
import json
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt



username = "rduan"
robot_id = 1
table_id = 1
camera_id = 1

headers = { "charset" : "utf-8", "Content-Type": "application/json" }

# def register_user(name=None):
#     global robot_id, table_id, camera_id, username

#     if name != None:
#         username = '%s' % name
        
#     conn = httpc.HTTPConnection("cerlab29.andrew.cmu.edu")
#     pdata = { "name" : username }
#     jdata = json.dumps(pdata, ensure_ascii = 'False')
#     conn.request("POST", "/IoRT/php/arm_reg_name.php", jdata, headers) # write to db
#     response = conn.getresponse()
#     #print(response.read().decode())
#     pdata = json.loads(response.read().decode())
#     robot_id = int(pdata['r_id'])
#     table_id = int(pdata['t_id'])
#     camera_id = int(pdata['c_id'])
#     username = '%s' % pdata['r_name']
#     print(username, robot_id, table_id, camera_id)
#     return pdata['ret']

# def check_camera_status():
#     global camera_id
#     conn = httpc.HTTPConnection("cerlab29.andrew.cmu.edu")
#     pdata = { "c_id" : camera_id }
#     jdata = json.dumps(pdata, ensure_ascii = 'False')
#     #print(jdata)
#     conn.request("POST", "/IoRT/php/camera_stat_r.php", jdata, headers) # write to db
#     response = conn.getresponse()
#     #print(response.read().decode())
#     pdata = json.loads(response.read().decode())
#     #print(pdata)
#     return pdata

def read_camera_2d(camera, timestamp, c_key=None):
    conn = httpc.HTTPConnection("cerlab29.andrew.cmu.edu")
    if c_key is None:
        pdata = { "c_id" : camera,
	          "c_time" : timestamp }
    else:
        pdata = { "c_key" : c_key }
        
    jdata = json.dumps(pdata, ensure_ascii = 'False')
    #print(jdata)
    conn.request("POST", "/RSIoT-2018/rsiot07/image_read.php", jdata, headers) # read from db
    response = conn.getresponse()
    #print(response.read().decode())
    pdata = json.loads(response.read().decode())
    #print(pdata)
    return pdata['data']

def write_camera_2d(camera, aux, timestamp, file):
    conn = httpc.HTTPConnection("cerlab29.andrew.cmu.edu")
    pdata = { "c_id" : camera,
              "aux" : aux,
              "file" : file,
              "c_time" : timestamp }
    jdata = json.dumps(pdata, ensure_ascii = 'False')
    #print(jdata)
    conn.request("POST", "/RSIoT-2018/rsiot07/image_write.php", jdata, headers) # write db
    response = conn.getresponse()
    #print(response.read().decode())
    pdata = json.loads(response.read().decode())
    
    if pdata['ret']:
        url = 'http://cerlab29.andrew.cmu.edu/RSIoT-2018/rsiot07/image_write2.php'
     
        fn = os.path.basename(pdata['data']['c_url']);
        f = {'file': (fn, open(file, 'rb'))}
        r = rq.post(url, files=f)

        return pdata['data']
    else:
        return {}

def write_position(camera, aux, timestamp, file):
    conn = httpc.HTTPConnection("cerlab29.andrew.cmu.edu")
    pdata = { "c_id" : camera,
              "aux" : aux,
              "file" : file,
              "c_time" : timestamp }
    jdata = json.dumps(pdata, ensure_ascii = 'False')
    #print(jdata)
    conn.request("POST", "/RSIoT-2018/rsiot07/position_write.php", jdata, headers) # write db
    response = conn.getresponse()
    #print(response.read().decode())
    pdata = json.loads(response.read().decode())
    
    if pdata['ret']:
        url = 'http://cerlab29.andrew.cmu.edu/RSIoT-2018/rsiot07/position_write2.php'
     
        fn = os.path.basename(pdata['data']['t_url']);
        f = {'file': (fn, open(file, 'rb'))}
        r = rq.post(url, files=f)

        return pdata['data']
    else:
        return {}

def read_position(camera, timestamp, c_key=None):
    conn = httpc.HTTPConnection("cerlab29.andrew.cmu.edu")
    if c_key is None:
        pdata = { "c_id" : camera,
              "c_time" : timestamp }
    else:
        pdata = { "c_key" : c_key }
        
    jdata = json.dumps(pdata, ensure_ascii = 'False')
    #print(jdata)
    conn.request("POST", "/RSIoT-2018/rsiot07/position_read.php", jdata, headers) # read from db
    response = conn.getresponse()
    #print(response.read().decode())
    pdata = json.loads(response.read().decode())
    #print(pdata)
    return pdata['data']
