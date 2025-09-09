DJ_USER = "sambray"
import os
import datajoint as dj
import numpy as np

import spyglass.common as sgc
import spyglass.spikesorting as sgs

# os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
# print(os.curdir)
import sys

sys.path.append("/home/sambray/Documents/MS_analysis_samsplaying/")
from ms_opto_stim_protocol import OptoStimProtocol
from Analysis.lfp_analysis import get_ref_electrode_index
from spyglass.common import Session, PositionIntervalMap

from spyglass.spikesorting import spikesorting_pipeline_populator, SortGroup

import multiprocessing

# ignore datajoint+jupyter async warnings
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.simplefilter("ignore", category=UserWarning)

import os
import signal

# Get the list of all process IDs currently running
pids = [pid for pid in os.listdir("/proc") if pid.isdigit()]

for pid in pids:
    try:
        # Get the command used to run each process
        cmd = open(os.path.join("/proc", pid, "cmdline"), "rb").read()
        # print(cmd)
        # If a specific keyword from your script is in the cmd, kill the process
        if b"spike_populator_parallel.py" in cmd:
            # os.kill(int(pid), signal.SIGKILL)
            print(cmd)
    except Exception:
        continue


"""
Generate the Call list for the parallel processing of the spike sorting pipeline
"""
call_list = []
sessions = [
    # "Winnie20220714_.nwb",
    # "Winnie20220715_.nwb",
    # "Winnie20220717_.nwb",
    # "Winnie20220719_.nwb",
    # "Winnie20220720_.nwb",
    #
    "Totoro20220602_.nwb",
    "Totoro20220603_.nwb",
    "Totoro20220607_.nwb",
    "Totoro20220614_.nwb",
    "Totoro20220615_.nwb",
    #
    "Yoshi20220509_.nwb",
    "Yoshi20220510_.nwb",
    "Yoshi20220513_.nwb",
    "Yoshi20220516_.nwb",
]

sessions = sessions + [
    "Yoshi20220517_.nwb",
    "Yoshi20220518_.nwb",
    #
    "Olive20220627_.nwb",
    "Olive20220628_.nwb",
    "Olive20220629_.nwb",
    "Olive20220707_.nwb",
    "Olive20220711_.nwb",
    #
    "Wallie20220911_.nwb",
    "Wallie20220912_.nwb",
    # "Wallie20220914_.nwb",
    "Wallie20220916_.nwb",
    "Wallie20220922_.nwb",
    #
    # "Dan20211112_.nwb",
    # "Dan20211113_.nwb",
    # "Dan20211114_.nwb",
]

sessions2 = [
    "Winnie20220714_.nwb",
    "Winnie20220715_.nwb",
    "Winnie20220717_.nwb",
    "Winnie20220719_.nwb",
    "Winnie20220720_.nwb",
    "Frodo20230808_.nwb",
    "Frodo20230809_.nwb",
    "Frodo20230810_.nwb",
    "Frodo20230811_.nwb",
    "Frodo20230814_.nwb",
    "Frodo20230815_.nwb",
    "Bilbo20230724_.nwb",
    "Bilbo20230725_.nwb",
    "Bilbo20230726_.nwb",
    "Bilbo20230802_.nwb",
    "Bilbo20230803_.nwb",
]
sessions = sessions + sessions2

from spyglass.common import PositionIntervalMap

for nwb_file_name in sessions:
    if not nwb_file_name in Session.fetch("nwb_file_name"):
        continue
    pos_intervals = (
        (OptoStimProtocol() * Session)
        & {
            # "subject_id": "Winnie",
            # "pulse_length_ms": 40,
            "nwb_file_name": nwb_file_name,
        }
    ).fetch("interval_list_name")
    pos_intervals = list(
        set(
            (PositionIntervalMap & {"nwb_file_name": nwb_file_name}).fetch(
                "position_interval_name"
            )
        )
    )
    intervals = [
        (
            PositionIntervalMap
            & {
                "position_interval_name": pos_interval,
                "nwb_file_name": nwb_file_name,
            }
        ).fetch1("interval_list_name")
        for pos_interval in pos_intervals
    ]

    team_name = "ms_stim"
    fig_url_repo = "gh://LorenFrankLab/sorting-curations/main/sambray/"
    sort_interval_name = None

    # make sort groups only if not currently available
    if len(sgs.SortGroup() & {"nwb_file_name": nwb_file_name}) == 0:
        print(nwb_file_name)
        ref_electrode = get_ref_electrode_index({"nwb_file_name": nwb_file_name})[0]
        ref_electrode_dict = {}
        for i in range(100):
            ref_electrode_dict[str(i)] = ref_electrode

        sgs.SortGroup().set_group_by_shank(
            nwb_file_name=nwb_file_name,
            references=ref_electrode_dict,
            omit_ref_electrode_group=True,
        )

    for interval in intervals:
        # if (
        #     len(
        #         sgs.CuratedSpikeSorting()
        #         & {
        #             "nwb_file_name": nwb_file_name,
        #             "sort_interval_name": sort_interval_name,
        #         }
        #     )
        #     == 0
        # ):
        call_list.append(
            (
                nwb_file_name,
                interval,
                team_name,
                fig_url_repo,
                sort_interval_name,
            )
        )
print(call_list)
# print(sgs.CuratedSpikeSorting())
# exit()


def pass_function(args):
    DJ_USER = "sambray"
    try:
        print(args[0], args[1])
        spikesorting_pipeline_populator(
            nwb_file_name=args[0],
            interval_list_name=args[1],
            team_name=args[2],
            fig_url_repo=args[3],
            sort_interval_name=args[4],
        )
        return
    except Exception as e:
        print("FAILED: ", args[0], args[1])
        print(e)
        return


# exit()
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.


class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""

            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc


try:
    pool = NonDaemonPool(processes=20)
    # pool = multiprocessing.Pool(processes=10)
    print("BEGINNING PARALLEL PROCESSING")
    pool.map(pass_function, call_list)
except:
    pool.close()
    pool.terminate()

# print("COMPLETE")
