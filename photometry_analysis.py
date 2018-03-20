import numpy as np
from nptdms import TdmsFile
import pylab as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from itertools import zip_longest
from scipy.stats import sem

# Importing data -----------------------------------------------------------------------

def load_tdms(pdata_filepath, smooth_SD=25):
    '''Open labview file and extract data, smooth fluorescence signals with gaussian filter
    of standard deviation smooth_SD ms.'''
    tdms_file = TdmsFile(pdata_filepath)
    time = 1000*tdms_file.object('Untitled', 'GCAMP').time_track()
    gcamp = tdms_file.object('Untitled', 'GCAMP').data             
    rfp = tdms_file.object('Untitled', 'RFP').data                 
    TTL = tdms_file.object('Untitled', 'pyControl').data
    start_time = tdms_file.object('Untitled', 'Time').data[0]      
    TTL_high = TTL>1.5
    TTL_times = time[np.where(TTL_high[1:] & ~TTL_high[:-1])[0]+1] 
    fs = 1000/np.median(time[1:]-time[:-1])
    if smooth_SD:
        gcamp = gaussian_filter1d(gcamp, smooth_SD*fs/1000.)
        rfp   = gaussian_filter1d(rfp  , smooth_SD*fs/1000.)

    return {'time_pho'  : time,       # Time since aquisition start (ms).
            'gcamp'     : gcamp,      # GCaMP signal (Volts).
            'rfp'       : rfp,        # RFP signal (Volts).
            'TTL_times' : TTL_times,  # Times when TTL signal went high (ms).
            'start_time': start_time, # Absolute time of first sample (datetime object).
            'fs'        : fs}         # Sampling rate (Hz) 

def add_data_to_session(session,  pdata_filepath):
    session.pdata = load_tdms(pdata_filepath)
    align_times(session)

def align_times(session):
    '''Align times of photometry data with that of pyControl data using the timestamps of 
    reward delivery recorded by both systems.'''
    # Work out which reward timestamps correspond to each other using cross correlation.
    ref_times_pyc = session.ordered_times(['reward_left', 'reward_right']) # Times of reward delivery as measerued by pyControl
    ref_times_pho = session.pdata['TTL_times'] # Times of reward delivery as measured by photometry system.
    dt_pyc = np.diff(ref_times_pyc) # Time differences between subsequent rewards (pyControl).
    dt_pho = np.diff(ref_times_pho) # Time differences between subsequent rewards (photometry).
    lag = np.argmax(np.correlate(dt_pyc-np.mean(dt_pyc),
                                 dt_pho-np.mean(dt_pho), mode='full'))-len(dt_pho)+1
    if lag == 0:
        l = min(len(ref_times_pyc), len(ref_times_pho))
        ref_times_pyc, ref_times_pho = (ref_times_pyc[:l], ref_times_pho[:l])    
    elif lag > 0:
        l = min(len(ref_times_pyc[lag:]), len(ref_times_pho))
        ref_times_pyc, ref_times_pho = (ref_times_pyc[lag:l+lag], ref_times_pho[:l])  
    elif lag < 0:
        l = min(len(ref_times_pyc), len(ref_times_pho[-lag:]))
        ref_times_pyc, ref_times_pho = (ref_times_pyc[:l], ref_times_pho[-lag:l-lag])
    max_rel_disc = np.max(np.abs((np.diff(ref_times_pyc)-
                                  np.diff(ref_times_pho))/np.diff(ref_times_pho)))
    if max_rel_disc > 0.001:
        print('Maximum relative discrepancy in time delta: {:.1}'.format(max_rel_disc))
    interpolate_func = interp1d(ref_times_pho, ref_times_pyc, fill_value='extrapolate')
    session.pdata['time_pyc'] = interpolate_func(session.pdata['time_pho'])

# Plotting --------------------------------------------------------------------------------

def mean_trace(sessions, ev_names, win_dur=[-1000,4000], fig_no=1, ebars='sem'):
    if not type(sessions) == list: sessions = [sessions]
    if not type(ev_names) == list: ev_names = [ev_names]
    traces = []
    for session in sessions:
        ev_times = session.ordered_times(ev_names)
        traces.append(traces_round_events(ev_times, session.pdata, win_dur))
    traces = np.vstack(traces)
    t = np.arange(traces.shape[1])*(1000./session.pdata['fs'])+win_dur[0]
    traces = traces - np.mean(traces[:,t<0],1)[:,None]
    mean_trace = np.mean(traces,0)
    std_trace = np.std(traces,0)
    sem_trace = sem(traces,0)
    if fig_no:
        plt.figure(fig_no, figsize=[4,4]).clf()
    plt.plot(t,mean_trace)
    if ebars:
        yerr = sem_trace if ebars == 'sem' else std_trace
        plt.fill_between(t, mean_trace-yerr, mean_trace+yerr, alpha=0.2, facecolor='b')
    plt.xlim(t[0],t[-1])
    plt.ylabel('GCaMP Fluorescence (AU)')
    plt.xlabel('Time (ms)')
    plt.tight_layout()


def patch_reward_responses(sessions, fig_no=1, win_dur=[-200,4000], n_rew_to_plot=6):
    '''Plot the reward response as a function of reward position in patch.'''
    traces_by_reward_n = []
    for session in sessions:
        patch_lengths = [len(patch['forage_time']) for patch in session.patch_data]
        patch_start_rewards = np.cumsum([0] + patch_lengths[:-1])
        patch_end_rewards = np.cumsum(patch_lengths)
        all_rewards = [ev for ev in session.events if ev.name in 
                         ['reward_left_available','reward_right_available']]
        patches = [all_rewards[ps:pe] for ps,pe in  zip(patch_start_rewards, patch_end_rewards)]
        valid_patches = []
        prev_reward_name = None
        for patch in patches: 
        # Reject invalid patches (due to bug in task there are occasionally patches with 
        # no rewards, such that patch before and after are on same side.
            if (len(patch) > 0 and     # Patch is not zero length.
               (len(set([r.name for r in patch])) == 1) and # All rewards in patch on same side
               (patch[0].name != prev_reward_name)):  # Rewards on different side from previous patch.
                valid_patches.append([ev.time for ev in patch])
                prev_reward_name = patch[0].name
            else:
                print('Invalid patch')
        patch_reward_times = np.array(list(zip_longest(*valid_patches, fillvalue=np.nan))).T
        for rew_n in range(patch_reward_times.shape[1]):
            reward_times = patch_reward_times[:,rew_n]
            reward_times = reward_times[~np.isnan(reward_times)]
            traces = traces_round_events(reward_times, session.pdata, win_dur)
            try:
                traces_by_reward_n[rew_n].append(traces)
            except IndexError:
                traces_by_reward_n.append([traces])
    traces_by_reward_n = [np.vstack(traces) for traces in traces_by_reward_n]
    t = np.arange(traces_by_reward_n[0].shape[1])*(1000./session.pdata['fs'])+win_dur[0]
    plt.figure(fig_no, figsize=[4,4]).clf()
    cm = plt.get_cmap('gist_rainbow')
    ax =plt.subplot(111)
    ax.set_color_cycle([cm(1.*i/n_rew_to_plot) for i in range(n_rew_to_plot)])
    for i, traces in enumerate(traces_by_reward_n[:n_rew_to_plot]):
        mean_trace = np.mean(traces,0)
        mean_trace -= np.mean(mean_trace[t<0])
        plt.plot(t,mean_trace, label='Reward {}'.format(i+1))
    plt.ylabel('GCaMP Fluorescence (AU)')
    plt.xlabel('Time relative to reward cue (ms)')
    plt.legend()
    plt.tight_layout()


def give_up_reward_responses(sessions, win_dur=[-1000,4000], fig_no=1):
    '''Compare reward responses for rewards where the subject did and did no give up on the
    patch immediately after recieving the reward.'''
    give_up_traces = []        # Traces around rewards where subject gave up patch imediately after.
    give_up_pos_in_patch = []  # Position of reward in patch for give up rewards (0 indexed).
    all_traces = []            # Traces round all rewards.
    all_pos_in_patch = []      # Position of reward in patch for all rewards (0 indexed).
    for session in sessions:
        rewards_per_patch = np.array([len(p['ave_time'])for p in session.patch_data])
        give_up_patches = np.where(np.array([p['give_up_time']  # Patches where subject gave up after reward.
                                   for p in session.patch_data]) == 0)
        give_up_reward_inds = np.cumsum(rewards_per_patch)[give_up_patches]-1
        reward_times = np.array([ev.time for ev in session.events 
                        if ev.name in ['reward_left_available', 'reward_right_available']])
        give_up_reward_times = reward_times[give_up_reward_inds]
        give_up_pos_in_patch.append(rewards_per_patch[give_up_patches]-1)
        give_up_traces.append(traces_round_events(give_up_reward_times, session.pdata, win_dur))
        ses_all_pos_in_patch = np.hstack([np.arange(rpp) for rpp in rewards_per_patch])
        ses_all_traces, valid_inds = traces_round_events(reward_times, session.pdata, win_dur, True)
        all_traces.append(ses_all_traces)
        all_pos_in_patch.append(ses_all_pos_in_patch[valid_inds])
    give_up_pos_in_patch = np.hstack(give_up_pos_in_patch)
    give_up_traces = np.vstack(give_up_traces)
    all_pos_in_patch = np.hstack(all_pos_in_patch)
    all_traces = np.vstack(all_traces)
    # Evaluate weighting of all reward traces to give equal influence of rewards at
    # different positions in patch for all rewards and give up rewards averages.
    max_rew_per_patch = max(all_pos_in_patch)
    n_give_up_rewards = give_up_pos_in_patch.shape[0]
    n_rewards = all_pos_in_patch.shape[0]
    all_rew_hist = np.histogram(all_pos_in_patch,np.arange(max_rew_per_patch+2))[0]
    give_up_hist = np.histogram(give_up_pos_in_patch,np.arange(max_rew_per_patch+2))[0]
    weights = (give_up_hist/n_give_up_rewards) / (all_rew_hist/n_rewards)
    # Plotting
    t = np.arange(all_traces.shape[1])*(1000./session.pdata['fs'])+win_dur[0]
    give_up_traces = give_up_traces - np.mean(give_up_traces[:,t<0],1)[:,None]
    mean_trace_give_up = np.mean(give_up_traces,0)
    sem_give_up_trace = sem(give_up_traces,0)
    mean_trace_all = np.mean(all_traces*weights[all_pos_in_patch][:,None],0)
    mean_trace_all -= np.mean(mean_trace_all[t<0])
    plt.figure(fig_no, figsize=[4,4]).clf()
    plt.plot(t, mean_trace_all, label='All rewards')  
    plt.fill_between(t, mean_trace_give_up-sem_give_up_trace, 
                        mean_trace_give_up+sem_give_up_trace, alpha=0.2, facecolor='r')
    plt.plot(t, mean_trace_give_up, 'r', label='Give up rewards')
    plt.xlim(t[0],t[-1])
    plt.ylabel('GCaMP Fluorescence (AU)')
    plt.xlabel('Time relative to reward cue (ms)')
    plt.legend()
    plt.tight_layout()



# Utility functions -----------------------------------------------------------------------

def traces_round_events(ev_times, pdata, win_dur=[-1000,4000], return_valid=False):
    '''Return an array of calcium traces round event times specified by ev_times argument'''
    pps = (np.array(win_dur)*pdata['fs']/1000).astype(int) # pre and post samples.
    ev_inds = np.argmax(pdata['time_pyc'] > ev_times[:,None], 1)
    valid_inds = (ev_inds>-pps[0]) & (ev_inds<(len(pdata['gcamp'])-pps[1]))
    ev_inds = ev_inds[valid_inds]
    if len(ev_inds) > 0:
        traces = np.vstack([pdata['gcamp'][i+pps[0]:i+pps[1]] for i in ev_inds])
    else:
        traces = np.zeros([0,-pps[0]+pps[1]])
    if return_valid:
        return traces, valid_inds
    else:
        return traces



