
# Stochastic COVID-19 Simulation on Networks
[![Build Status](https://travis-ci.com/gerritgr/StochasticCovid19.svg?token=qQ7vTmAySdBppYxywojC&branch=master)](https://travis-ci.com/gerritgr/StochasticCovid19)

Copyright: 2020, [Gerrit Großmann](https://mosi.uni-saarland.de/people/gerrit/), [Group of Modeling and Simulation](https://mosi.uni-saarland.de/) at [Saarland University](http://www.cs.uni-saarland.de/)

Version: 0.1 (Please note that this is proof-of-concept code in a very early development stage.)

**Caveat lector**: This is an academic model, do not use academic models as a basis for political decision-making.

## Overview
------------------
![Animation](https://github.com/gerritgr/StochasticCovid19/raw/master/anim-opt.gif)
![Lineplot](https://github.com/gerritgr/StochasticCovid19/raw/master/output_geom.jpg)

Stochastic (Monte-Carlo) simulation of the of Covid-19 pandemic (of the SARS-CoV-2 virus) on complex networks (contact graphs).
The model falls under the general class of a SEIR compartment models.  
The model is a stochastic interpretation of a Covid-19 model based on [the work of Dr. Alison Hill](https://alhill.shinyapps.io/COVID19seir/).


## Installation
------------------
With:
```console
pip install -r requirements.txt
```
## Example Usage
-----------------
With
```console
python simulation.py
```
