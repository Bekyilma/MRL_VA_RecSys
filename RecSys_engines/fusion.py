#!/usr/bin/env python3
# coding: utf-8

"""
API server for fusion-based RecSys engines.

Authors:
- Bereket A. Yilma <name.surname@uni.lu>
- Luis A. Leiva <name.surname@uni.lu>
"""

import sys
from flask import Flask, request, jsonify
from waitress import serve
from lda_engine import LDAEngine
from bert_engine import BertEngine
from resnet_engine import ResNetEngine

lda = LDAEngine()
bert = BertEngine()
resnet = ResNetEngine()

# CLI argument to choose fusion model and type.
fusion_id = sys.argv[1] if len(sys.argv) > 1 else "lda_resnet"
fusion_type = sys.argv[2] if len(sys.argv) > 2 else "partial"

if fusion_type == "partial":
    from fusion_engine_partial import FusionEnginePartial as FusionEngine
else:
    from fusion_engine_total import FusionEngineTotal as FusionEngine

# Specify port for fusion engines.
if fusion_id == "lda_resnet":
    eng = FusionEngine(models=(lda, resnet), model_weights=(0.50, 0.50))
    if fusion_type == "partial":
        port = 10501
    else:
        port = 10502

elif fusion_id == "lda_bert":
    eng = FusionEngine(models=(lda, bert), model_weights=(0.50, 0.50))
    if fusion_type == "partial":
        port = 10503
    else:
        port = 10504

elif fusion_id == "bert_resnet":
    eng = FusionEngine(models=(bert, resnet), model_weights=(0.50, 0.50))
    if fusion_type == "partial":
        port = 10505
    else:
        port = 10506

else:
    raise ValueError(f'Fusion "{fusion_id}" not understood. Try e.g. "lda_resnet" or "bert_resnet"')


app = Flask(fusion_id)

@app.route("/retrieval", methods=["POST"])
def retrieval():
    """
    preferences = {"000-03W3-0000": 5, "000-02UP-0000": 5, "000-03IY-0000": 5, "000-0344-0000": 5, "000-02XU-0000": 5,
                   "000-01DN-0000": 5, "000-04PW-0000": 5, "000-017I-0000": 5, "000-04S4-0000": 5}
    """
    preferences = request.json

    recommendations = eng.retrieval(preferences, n=9)
    return jsonify(recommendations)


# Use a production server instead of Flask's built-in one.
# app.run(debug=False, port=port)
serve(app, port=port, threads=10)
