# DIG-SSL
### Code structure
```
ssl
 |- utils
 |   |- encoders.py
 |   |- datasets.py
 |
 |- predictive
 |- eval
 |   |- unsupervised
 |   |- semi_supervised
 |   |- ...
 |
 |- contrastive
     | 
     |— utils
     |     | — …
     | 
     |— model
     |     | — __init__.py
     |     | — contrastive.py
     |     | — infograph.py
     |     | — graphcl.py
     |     | — multiview.py
     |     | — gcc.py
     | 
     |— objectives
     |     | — __init__.py
     |     | — infonce.py
     |     | — jse.py
     | 
     |— views_fn
           | — __init__.py
           | — feature.py
           | — structure.py
           | — sample.py
 ```
