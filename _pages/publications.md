---
layout: archive
permalink: /publications/
title: "Publications"
author_profile: true
---

You can also find my publications on <u><a href="https://scholar.google.com/citations?user=im5fNaQAAAAJ&hl=en">my Google Scholar profile</a>.</u>

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}