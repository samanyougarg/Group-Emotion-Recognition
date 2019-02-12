#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import render_template, jsonify, current_app, request
from app.models.building import BuildingModel
from . import main
from app import db

@main.route('/', methods=['GET', 'POST'])
def index():
    return render_template('main/index.html')
