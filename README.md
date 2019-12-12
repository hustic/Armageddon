# ACSE-4-armageddon

Asteroids entering Earth’s atmosphere are subject to extreme drag forces that decelerate, heat and disrupt the space rocks. The fate of an asteroid is a complex function of its initial mass, speed, trajectory angle and internal strength. 

[Asteroids](https://en.wikipedia.org/wiki/Asteroid) 10-100 m in diameter can penetrate deep into Earth’s atmosphere and disrupt catastrophically, generating an atmospheric disturbance ([airburst](https://en.wikipedia.org/wiki/Air_burst)) that can cause [damage on the ground](https://www.youtube.com/watch?v=tq02C_3FvFo). Such an event occurred over the city of [Chelyabinsk](https://en.wikipedia.org/wiki/Chelyabinsk_meteor) in Russia, in 2013, releasing energy equivalent to about 520 [kilotons of TNT](https://en.wikipedia.org/wiki/TNT_equivalent) (1 kt TNT is equivalent to 4.184e12 J), and injuring thousands of people ([Popova et al., 2013](http://doi.org/10.1126/science.1242642); [Brown et al., 2013](http://doi.org/10.1038/nature12741)). An even larger event occurred over [Tunguska](https://en.wikipedia.org/wiki/Tunguska_event), an unpopulated area in Siberia, in 1908. 

This tool predicts the fate of asteroids entering Earth’s atmosphere for the purposes of hazard assessment.

### Installation Guide

To install the module using pip, please run
```
pip install git+https://github.com/acse-2019/acse-4-armageddon-mathilde.git
```

### User instructions

The module can be imported with
```
>>> import armageddon
```

#### Planet
The core functionality is to simulate an asteroid entering the atmosphere under specified initial conditions. 
This functionality can be called in the following example format:
```
>>> planet = armageddon.Planet(atmos_func='exponential')
>>> results, outcomes = planet.impact(radius=10,velocity=2.1e4,density=3e3,strength=1e5,angle=45)
```
Where the specified parameters can be changed as desired. This outputs a pandas DataFrame of the parameters and loss of kinetic energy at each timestep, and a dictionary containing an analysis of the scenario.
Please refer to the [documentation](./docs_build/index.html) in the `index.html` file under the `docs_build` directory for more information about choosing input parameters.
After running the simulation, some basic plots can be generated using:
```
>>> insert text here
```

#### Ensemble
In addition to the core functionality, it is possible to perform an ensemble of simulations to vary specified input parameters according to their respective probability distributions and find the distribution of corresponding burst altitudes.
This can be done in the following format:
```
>>> planet = armageddon.Planet(atmos_func='exponential')
>>> fiducial_impact = {'radius': 10.0,
                       'angle': 45.0,
'strength': 100000.0,
                       'velocity': 21000.0,
                       'density': 3000.0}
>>> ensemble = armageddon.ensemble.solve_ensemble(planet,fiducial_impact,variables=[],rmin=8,rmax=12)
```
Where the parameters and fiducial values can be specified, as well as the input variables to be varied.
For more information regarding the use of this functionality, please refer to the [documentation](./docs_build/index.html).

### Documentation

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be built by running

```
python -m sphinx docs html
```

then viewing the `index.html` file in the `docs_build` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `Armageddon.tex` file in the `latex` directory in your browser.

### Testing

The tool includes a fully automated testing suite, which you can use to check its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```
python -m pytest armageddon
```
