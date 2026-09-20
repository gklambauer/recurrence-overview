---
layout: default
title: "The Illustrated Recurrence: From Amari-Hopfield Nets to GPT-6 Astra"
mathjax: true
---

{% capture readme %}{% include_relative README.md %}{% endcapture %}
{% capture open_math %}

$${% endcapture %}
{% capture close_math %}$$

{% endcapture %}
{{ readme | replace: "```math", open_math | replace: "```", close_math }}
