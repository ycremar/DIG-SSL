# DIG-SSL
### Code structure
```
ssl
 |- utils (JT)
 |   |- encoders.py
 |   |- datasets.py
 |
 |- predictive
 |- eval (JT)
 |   |- unsupervised
 |   |- semi_supervised
 |   |- ...
 |
 |- contrastive
     | 
     |— utils
     |     | — …
     | 
     |— model (YC)
     |     | — __init__.py
     |     | — contrastive.py
     |     | — infograph.py
     |     | — graphcl.py
     |     | — multiview.py
     |     | — gcc.py
     | 
     |— objectives 
     |     | — __init__.py
     |     | — infonce.py (YC, done)
     |     | — jse.py (Zhao)
     | 
     |— views_fn (Zhao)
           | — __init__.py
           | — feature.py
           | — structure.py
           | — sample.py
 ```
