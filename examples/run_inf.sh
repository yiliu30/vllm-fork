# HABANA_PROFILE_WRITE_HLTV=1 HABANA_PROFILE=1
# hl-prof-config --use-template profile_api_with_nics --fuser on --trace-analyzer on --trace-analyzer-xlsx on
# hl-prof-config --gaudi2
# hl-prof-config --gaudi3

hl-prof-config   -o  ./new_trace_results

# !!!!!!!!!!!! ENABLE GRAPH_VISUALIZATION, and dowdload the ./graph_dump to local,
# open the hltv
# we can open it in perfetto


GRAPH_VISUALIZATION=1 \
VLLM_PROFILE_EXECUTE_MODEL_DECODE=1 \
VLLM_PROFILE_EXECUTE_MODEL_PROMPT=1 \
HABANA_PROFILE=1 \
HABANA_PROFILE_WRITE_HLTV=1 \
VLLM_PROFILE_EXECUTE_MODEL_PROMPT=1 \
VLLM_PROFILE_EXECUTE_MODEL_DECODE_STEPS=5 \
python  offline_inference.py



# root@3FG5:/mnt/disk3/yiliu4/vllm-fork/examples# cat /root/.habana/prof_config.json 
# {
#     "GeneralSettings": {
#         "values": {
#             "outdir": {
#                 "value": "./trace_results"
#             }
#         }
#     },
#     "Plugins": [
#         {
#             "enable": true,
#             "lib": "libhost_profiler.so",
#             "name": "HostProfiler",
#             "values": {
#                 "start_disabled": {
#                     "value": true
#                 }
#             }
#         },
#         {
#             "enable": true,
#             "lib": "libhw_trace.so",
#             "name": "HwTrace",
#             "values": {
#                 "archProfileUnits": {
#                     "gaudi": {
#                         "NIC": {
#                             "NIC0_0": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC0_1": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC1_0": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC1_1": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC2_0": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC2_1": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC3_0": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC3_1": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC4_0": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "NIC4_1": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "enable": {
#                                 "value": true
#                             }
#                         }
#                     },
#                     "gaudi2": {
#                         "NIC": {
#                             "enable": {
#                                 "value": true
#                             }
#                         },
#                         "ROT": {
#                             "ROT0_CS": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "ROT1_CS": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "enable": {
#                                 "value": true
#                             }
#                         },
#                         "VDEC": {
#                             "DurationEvents": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "enable": {
#                                 "value": true
#                             }
#                         }
#                     },
#                     "gaudi3": {
#                         "CS": {
#                             "Gaudi3CSAdvancedProfiling": {
#                                 "value": 0
#                             },
#                             "enable": {
#                                 "value": false
#                             }
#                         },
#                         "NIC": {
#                             "D0_NIC0_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D0_NIC1_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D0_NIC2_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D0_NIC3_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D0_NIC4_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D0_NIC5_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D1_NIC0_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D1_NIC1_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D1_NIC2_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D1_NIC3_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D1_NIC4_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "D1_NIC5_CS_DBG": {
#                                 "enable": {
#                                     "value": true
#                                 }
#                             },
#                             "enable": {
#                                 "value": true
#                             }
#                         }
#                     }
#                 },
#                 "generalOptions": {
#                     "arch": {
#                         "value": "gaudi3"
#                     },
#                     "profilePhase": {
#                         "value": "profileApi"
#                     },
#                     "traceBufferSize": {
#                         "value": "0x8000000"
#                     }
#                 },
#                 "parseOptions": {
#                     "addFuserMetadata": {
#                         "value": true
#                     },
#                     "traceAnalyzer": {
#                         "traceAnalyzerJson": {
#                             "value": true
#                         },
#                         "traceAnalyzerXlsx": {
#                             "value": true
#                         }
#                     }
#                 }
#             }
#         }
#     ]