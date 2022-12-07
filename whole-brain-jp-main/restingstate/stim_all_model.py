#!/usr/bin/env python
# -*- coding: utf-8 -*-
#please set all flags before run the simulation
##

import fetch_params
import ini_all
import nest_routine
import nest
import nest.topology as ntop
import numpy as np
import time
import sys
import roslibpy
import glob

def main():
    # 1) reads parameters
    sys.stdout.write('#1\n')
    sys.stdout.flush()
    sim_params = fetch_params.read_sim()

    sys.stdout.write('dt=')
    sys.stdout.write(str(sim_params['dt']))
    sys.stdout.write('...')
    sys.stdout.flush()
    
    # simParams.pyでonになってるものをシミュレーションする（restingstate）
    for sim_model in sim_params['sim_model'].keys():
        if sim_params['sim_model'][sim_model]['on']:
            sim_regions=sim_params['sim_model'][sim_model]['regions']
            sim_model_on=sim_model
            print ('simulation model ', sim_model_on, ' will start')
    # 各脳部位がTrueならば、パラメータを読み込む
    if sim_regions['S1']:
        ctx_params = fetch_params.read_ctx()
    if sim_regions['M1']:
        ctx_M1_params = fetch_params.read_ctx_M1()
    if sim_regions['TH_S1'] or sim_regions['TH_M1']:
        th_params = fetch_params.read_th()
    if sim_regions['BG']:
        bg_params = fetch_params.read_bg()
    if sim_regions['CB_S1'] or sim_regions['CB_M1']:
        cb_params = fetch_params.read_cb()
    conn_params = fetch_params.read_conn()

    # 1.5) initialize nest
    sys.stdout.write('#1.5\n')
    sys.stdout.flush()
    # nestの細かい設定をやっている
    nest_routine.initialize_nest(sim_params)
    wb_layers = {}
    # 2) instantiates regions
    sys.stdout.write('#2\n')
    sys.stdout.flush()
    
    sys.stdout.write('#S1\n')
    sys.stdout.flush()
    if sim_regions['S1']:
        # ctxの定義
        ctx_layers, S1_L4_Pyr = ini_all.instantiate_ctx(ctx_params, sim_params['scalefactor'], sim_params['initial_ignore'])
        wb_layers = dict(wb_layers, **ctx_layers)
    sys.stdout.write('#M1\n')
    sys.stdout.flush()
    if sim_regions['M1']:
        # M1の定義
        ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'],sim_params['initial_ignore'])
        wb_layers = dict(wb_layers, **ctx_M1_layers)
    sys.stdout.write('#TH_S1\n')
    sys.stdout.flush()
    if sim_regions['TH_S1'] or sim_regions['TH_M1']:
        # THの定義
        th_layers = ini_all.instantiate_th(th_params, sim_params['scalefactor'],sim_params['initial_ignore'])
        wb_layers = dict(wb_layers, **th_layers)
    sys.stdout.write('#CB_S1\n')
    sys.stdout.flush()
    if sim_regions['CB_S1']:
        # cb
        cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'], sim_params)
        wb_layers = dict(wb_layers, **cb_layers_S1)
    sys.stdout.write('#CB_M1\n')
    sys.stdout.flush()
    if sim_regions['CB_M1']:
        cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'], sim_params)
        wb_layers = dict(wb_layers, **cb_layers_M1)

    sys.stdout.write('#BG\n')
    sys.stdout.flush()
    if sim_regions['BG']:
        if sim_params['channels']:
            bg_params['channels'] = True 
        else:
            bg_params['channels'] = False 
        bg_params['circle_center'] = nest_routine.get_channel_centers(sim_params, hex_center=[0, 0],
                                                                ci=sim_params['channels_nb'],
                                                                hex_radius=sim_params['hex_radius'])
        sys.stdout.write('#BG inter-regional connection\n')
        sys.stdout.flush()

        #### Basal ganglia inter-regional connection with S1 and M1 ######
        if sim_regions['S1']:
            if sim_regions['M1']:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': {'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
            else:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': None, #{'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': {'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
        else:
            if sim_regions['M1']:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': None, #{'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
            else:
                bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                               ctx_inputs={'M1': None, #{'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                           'S1': None, #{'layers': ctx_layers, 'params': ctx_params},
                                                           'M2': None},
                                                            scalefactor=sim_params['scalefactor'])
        wb_layers = dict(wb_layers, **bg_layers)

    sys.stdout.write('#BG inter-regional connection done\n')
    sys.stdout.flush()
    
    # 3) interconnect regions
    print('#3')
    start_time = time.time()
    if sim_regions['S1'] and sim_regions['M1']:
        pass
        # _ = nest_routine.connect_inter_regions('S1', 'M1')
    if sim_regions['S1'] and sim_regions['CB_S1']:
        _ = nest_routine.connect_inter_regions('S1', 'CB', conn_params, wb_layers)
    if sim_regions['M1'] and sim_regions['CB_M1']:
        _ = nest_routine.connect_inter_regions('M1', 'CB', conn_params, wb_layers)
    if sim_regions['S1'] and sim_regions['TH_S1']:
        _ = nest_routine.connect_inter_regions('S1', 'TH', conn_params, wb_layers)
        _ = nest_routine.connect_inter_regions('TH', 'S1', conn_params, wb_layers)
    if sim_regions['M1'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_inter_regions('M1', 'TH', conn_params, wb_layers)
        _ = nest_routine.connect_inter_regions('TH', 'M1', conn_params, wb_layers)
    if sim_regions['CB_S1'] and sim_regions['TH_S1']:
        _ = nest_routine.connect_inter_regions('CB', 'TH', conn_params, wb_layers)
    if sim_regions['CB_M1'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_inter_regions('CB', 'TH', conn_params, wb_layers)
    if sim_regions['BG'] and sim_regions['TH_M1']:
        _ = nest_routine.connect_inter_regions('BG', 'TH', conn_params, wb_layers)
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Interconnect_Regions_Time ' + str(time.time() - start_time) + '\n')

    ### adding corrections of edge effect (this mitigation works partially) ####
    _ = nest_routine.reduce_weights_at_edges(
            'M1_L5B_PT',
            'CB_M1_layer_pons',
            wb_layers['M1_L5B_PT'],
            wb_layers['CB_M1_layer_pons'],
            margin=0.025,
            new_weight=0.0
            )
    _ = nest_routine.reduce_weights_at_edges(
            'GPi_fake',
            'TH_M1_IZ_thalamic_nucleus_TC',
            wb_layers['GPi_fake'],
            wb_layers['TH_M1_IZ']['thalamic_nucleus_TC'],
            margin=0.025,
            new_weight=-1940.
            )



    # 2.5) detectors
    print('#2.5')
    detectors = {}

    # sys.stdout.write('#BG\n')
    # sys.stdout.flush()
    # if sim_regions['BG']:
    #     for layer_name in bg_layers.keys():
    #         detectors[layer_name] = nest_routine.layer_spike_detector(bg_layers[layer_name], layer_name,
    #                                                                   sim_params['initial_ignore'])
    # sys.stdout.write('#S1\n')
    # sys.stdout.flush()
    # if sim_regions['S1']:
    #     for layer_name in ctx_layers.keys():
    #         detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name, sim_params['initial_ignore'])

    sys.stdout.write('#M1\n')
    sys.stdout.flush()
    if sim_regions['M1']:
        for layer_name in ctx_M1_layers.keys():
            detectors[layer_name] = nest_routine.layer_spike_detector(
                    ctx_M1_layers[layer_name], 
                    layer_name, 
                    sim_params['initial_ignore']
                    )
    sys.stdout.write('#others\n')
    sys.stdout.flush()
    # sys.stdout.write('#CB_S1\n')
    # sys.stdout.flush()
    # if sim_regions['CB_S1']:
    #     for layer_name in cb_layers_S1.keys():
    #         detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_S1[layer_name], layer_name, sim_params['initial_ignore'])
    # sys.stdout.write('#CB_M1\n')
    # sys.stdout.flush()
    # if sim_regions['CB_M1']:
    #     for layer_name in cb_layers_M1.keys():
    #         detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_M1[layer_name], layer_name, sim_params['initial_ignore'])
    # sys.stdout.write('#TH_S1\n')
    # sys.stdout.flush()
    # if sim_regions['TH_S1']:
    #     sys.stdout.write('#TH_S1 in\n')
    #     sys.stdout.flush()
    #     sys.stdout.write('  #TH_S1_EZ\n')
    #     sys.stdout.flush()
    #     for layer_name in th_layers['TH_S1_EZ'].keys():
    #         detectors['TH_S1_EZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_EZ'][layer_name], 'TH_S1_EZ_'+layer_name, sim_params['initial_ignore'])
    #     sys.stdout.write('  #TH_S1_IZ\n')
    #     sys.stdout.flush()
    #     for layer_name in th_layers['TH_S1_IZ'].keys():
    #         detectors['TH_S1_IZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_IZ'][layer_name], 'TH_S1_IZ_'+layer_name, sim_params['initial_ignore'])
    # sys.stdout.write('#TH_M1\n')
    # sys.stdout.flush()
    # if sim_regions['TH_M1']:
    #     sys.stdout.write('#TH_M1 in\n')
    #     sys.stdout.flush()
    #     sys.stdout.write('  #TH_M1_EZ\n')
    #     sys.stdout.flush()
    #     for layer_name in th_layers['TH_M1_EZ'].keys():
    #         detectors['TH_M1_EZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_EZ'][layer_name], 'TH_M1_EZ_'+layer_name, sim_params['initial_ignore'])
    #     sys.stdout.write('  #TH_M1_IZ\n')
    #     sys.stdout.flush()
    #     for layer_name in th_layers['TH_M1_IZ'].keys():
    #         detectors['TH_M1_IZ_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_IZ'][layer_name], 'TH_M1_IZ_'+layer_name, sim_params['initial_ignore'])

    sys.stdout.write('sim_model_on\n')
    sys.stdout.flush()
    print (sim_model_on)
    if sim_model_on=='resting_state':
        sys.stdout.write('sim_model_on==resting_state\n')
        sys.stdout.flush()
        simulation_time = sim_params['simDuration']+sim_params['initial_ignore']
        print('Simulation Started:')
        start_time = time.time()
        sys.stdout.write('simulation_time=')
        sys.stdout.write(str(simulation_time))
        sys.stdout.write('start_time=')
        sys.stdout.write(str(start_time))
        sys.stdout.flush()
        sys.stdout.write('simulation start')
        
        IP = "192.168.12.134"
        nrp = NRP(IP)
        simulation_num = int(simulation_time / 100)
        fr1, fr2 = 0.0, 0.0
        for n in range(simulation_num):
            # rate is firing rate of S1 L4 Pyr
            # fr is firing rate of M1 L5B PT

            # translate the angle into poisson spike rate 
            rate1, rate2 = angle2rate()

            # connect to Pyr nuclei in S1 layer 4
            connect_to_S1_L4_Pyr(S1_L4_Pyr, 100.0*n, rate1, rate2)
            
            print("{} ms - {} ms".format(n*100.0, (n+1)*100.0))
            nest.Simulate(100.0)

            # publish the activation of PT nuclei in M1 layer 5B
            fr1, fr2, = calculate_fr(n)
            nrp.set_state((fr1 - fr2)/100)
            nrp.send_action()
            
        
        sys.stdout.write('simulation done')
        sys.stdout.flush()
        
        with open('./log/' + 'performance.txt', 'a') as file:
            file.write('Simulation_Elapse_Time ' + str(time.time() - start_time) + '\n')
        print ('Simulation Finish')


    #     if sim_regions['BG']:
    #         for layer_name in bg_layers.keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files(layer_name, detectors[layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(bg_layers[layer_name]))
    #             print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['S1']:
    #         for layer_name in ctx_layers.keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files(layer_name,detectors[layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(ctx_layers[layer_name]))
    #             print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['M1']:
    #         for layer_name in ctx_M1_layers.keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files(layer_name,detectors[layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(ctx_M1_layers[layer_name]))
    #             print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['CB_S1']:
    #         for layer_name in cb_layers_S1.keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files(layer_name,detectors[layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(cb_layers_S1[layer_name]))
    #             print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['CB_M1']:
    #         for layer_name in cb_layers_M1.keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files(layer_name, detectors[layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(cb_layers_M1[layer_name]))
    #             print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")

    #     if sim_regions['TH_S1']:
    #         for layer_name in th_layers['TH_S1_EZ'].keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files('TH_S1_EZ_' + layer_name, detectors['TH_S1_EZ_' + layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(th_layers['TH_S1_EZ'][layer_name]))
    #             print('Layer ' + 'TH_S1_EZ_' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['TH_S1']:
    #         for layer_name in th_layers['TH_S1_IZ'].keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files('TH_S1_IZ_' + layer_name, detectors['TH_S1_IZ_' + layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(th_layers['TH_S1_IZ'][layer_name]))
    #             print('Layer ' + 'TH_S1_IZ_' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['TH_M1']:
    #         for layer_name in th_layers['TH_M1_EZ'].keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files('TH_M1_EZ_' + layer_name, detectors['TH_M1_EZ_' + layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(th_layers['TH_M1_EZ'][layer_name]))
    #             print('Layer ' + 'TH_M1_EZ_' + layer_name + " fires at " + str(rate) + " Hz")
    #     if sim_regions['TH_M1']:
    #         for layer_name in th_layers['TH_M1_IZ'].keys():
    #             rate = nest_routine.get_firing_rate_from_gdf_files('TH_M1_IZ_' + layer_name, detectors['TH_M1_IZ_' + layer_name], sim_params['simDuration'],
    #                                            nest_routine.count_layer(th_layers['TH_M1_IZ'][layer_name]))
    #             print('Layer ' + 'TH_M1_IZ_' + layer_name + " fires at " + str(rate) + " Hz")


    # else:
    #     print ('wrong model set')

class NRP:
    def __init__(self, IP="192.168.12.134"):
        self.client = roslibpy.Ros(host=IP, port=9090)
        self.client.run()
        self.listener = roslibpy.Topic(
                client, 
                '/gazebo_muscle_interface/robot/muscle_states', 
                'gazebo_ros_muscle_interface/MuscleStates',
                queue_size=1)
        self.listener.subscribe(CallBack)

        muscle_names = ["Foot1", "Foot2", "Humerus1", "Humerus2", "Radius1", "Radius2"]
        self.talker = []
        for name in muscle_names:
            self.talker.append( 
                    roslibpy.Topic(
                        client, 
                        '/gazebo_muscle_interface/robot/' + name + '/cmd_activation', 'std_msgs/Float64') )
        self.state = 0.0

    def set_state(self, delta):
        self.state = self.state + delta
        self.state = 1 if self.state > 1 else -1 if self.state < -1 else self.state

    def send_action(self):
        bwd = [0., 1., 0., 1., 1., 0.]
        fwd = [1., 0., 1., 0., 0., 1.]
        if self.state >= 0:
            for i, t in enumerate(self.talker):
                t.publish( roslibpy.Message({ 'data': self.state * fwd[i]}) )
        elif self.state < 0:
            for i, t in enumerate(self.talker):
                t.publish( roslibpy.Message({ 'data': -self.state * bwd[i]}) )


def CallBack(msg):
    global angle
    muscles = msg['muscles']
    humerus2 = muscles[5]
    name = humerus2['name']
    path_points = humerus2['path_points']
    point1 = np.array([path_points[1]['x'], path_points[1]['y'], path_points[1]['z']])
    point2 = np.array([path_points[2]['x'], path_points[2]['y'], path_points[2]['z']])
    point3 = np.array([path_points[3]['x'], path_points[3]['y'], path_points[3]['z']])
    vec1 = point1 - point2
    vec2 = point3 - point2
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    inner = np.dot(vec1, vec2)
    angle = np.rad2deg(np.arccos(inner / (vec1_norm * vec2_norm)))



def connect_to_S1_L4_Pyr(layer, start_time, rate1, rate2):
    n = len(nest.GetNodes(layer)[0])
    for _ in range(0, 10):
        pg1 = nest.Create(
                "poisson_generator", 
                int(n/2), 
                params={"rate": rate1/10.0, "start": start_time, "stop": start_time + 100.0})
        nest.Connect(
                pre=pg1, 
                post=nest.GetNodes(layer)[0][:int(n/2)], 
                conn_spec='one_to_one', 
                syn_spec={'weight': 100.0})
        pg2 = nest.Create(
                "poisson_generator", 
                int(n/2), 
                params={"rate": rate2/10.0, "start": start_time, "stop": start_time + 100.0})
        nest.Connect(
                pre=pg2, 
                post=nest.GetNodes(layer)[0][int(n/2):], 
                conn_spec='one_to_one', 
                syn_spec={'weight': 100.0})

def angle2rate():
    # if angle is 1.4, fr1, fr2 are   0 Hz, 100 Hz
    # if angle is 2.8, fr1, fr2 are 100 HZ,   0 H
    rate1 = ((angle - 1.4) /  1.4) * 100.0
    rate2 = 101.1 - rate1
    return rate1, rate2

def calculate_fr(n):
    N = 5819
    N1 = int(N/2) + 1
    N2 = int(N/2)

    count1, count2 = 0, 0
    logfiles = glob.glob(os.path.join("log", "M1_L5B_PT-*"))
    for logfile in logfiles:
        #print(logfile)
        with open(logfile, "r") as f:
            for line in reversed(list(f)):
                toks = line.rstrip("\n").split("\t")
                if len(toks) != 3: continue
                if int(float(toks[1])/100) != n:
                    break
                neuron_id = int(toks[0])
                # pick up spike timing during recent 100ms
                if neuron_id <= 129678: # 直打ちしてすみません。
                    count1 += 1
                else:
                    count2 += 1
    fr1, fr2 = count1/(N1/0.1), count2/(N2/0.1)
    return fr1, fr2


if __name__ == '__main__':
    main()
