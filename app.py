import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import gradio as gr
import torch

TITLE = "SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation"
DESCRIPTION = '''
<style>
figure {
    margin: 0;
    font-size: smaller;
    text-align: justify;
}
img {
    width: auto;
    max-width: 100%;
    height: auto;
}
video {
    width: 720;
    max-width: 100%;
    height: 405;
}
ul.horizontal {
    padding: 0;
}
ul.horizontal li {
    padding: 0 1em 0 0;
    display: inline-block;
}
table td {
    vertical-align: top;
}
</style>

<table>
<tr>
<td>
<figure>
<img src="https://caizhongang.com/projects/SMPLer-X/assets/teaser_complete.png" alt="sketch2pose">
</figure>
<p>
<ul class="horizontal">
<li><a href="https://caizhongang.com/projects/SMPLer-X/">[project page]</a></li>
<li><a href="https://arxiv.org/pdf/2309.17448.pdf">[NeurIPS 2023 (Dataset and Benchmark Track)]</a></li>
<li><a href="https://github.com/caizhongang/SMPLer-X">[code.git]</a></li>
</ul>
</p>
</td>
<td>
<video width="720" height="405" controls autoplay muted loop>
<source src="https://www.youtube.com/watch?v=DepTqbPpVzY" type="video/mp4">
</video>
</td>
</tr>
<table>
<p>
The first generalist foundation model for expressive human pose and shape estimation (EHPS)
</p>
'''