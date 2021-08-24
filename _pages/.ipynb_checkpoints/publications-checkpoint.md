---
layout: archive
permalink: /publications/
title: "Publications"
author_profile: true
redirect_from: 
  - /wordpress/academic-papers/
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

# {% include base_path %}

# {% for post in site.publications reversed %}
#  {% include archive-single.html %}
# {% endfor %}




{% include base_path %}
{% capture written_year %}'None'{% endcapture %}
{% for post in site.publications %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% if year != written_year and post.visible != 'false' %}
    <h2 id="{{ year | slugify }}" class="archive__subtitle">{{ year }}</h2>
    {% capture written_year %}{{ year }}{% endcapture %}
  {% endif %}
  {% include archive-single.html %}
{% endfor %}
