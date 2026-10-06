/* CO5085 — scroll snapping, quick-nav sync and pagination dots. */
(function () {
  'use strict';

  var scroller = document.querySelector('.scroller');
  if (!scroller) return;

  var pages    = Array.prototype.slice.call(scroller.querySelectorAll('.page'));
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.quicknav a[href^="#"]'));
  var dotWrap  = document.querySelector('.dots');
  var bar      = document.querySelector('.progress');
  var reduced  = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- Build the pagination dots from the sections themselves ---- */
  var dots = [];
  if (dotWrap) {
    pages.forEach(function (page, i) {
      var label = page.getAttribute('data-title') || ('Trang ' + (i + 1));
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('aria-label', 'Đi tới: ' + label);
      b.innerHTML = '<span class="tip"></span>';
      b.querySelector('.tip').textContent = label;
      b.addEventListener('click', function () { goTo(i); });
      dotWrap.appendChild(b);
      dots.push(b);
    });
  }

  function goTo(i) {
    if (!pages[i]) return;
    pages[i].scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
  }

  /* ---- Mark one section active across nav, dots and the URL hash ---- */
  var current = -1;

  function setActive(i) {
    if (i === current || !pages[i]) return;
    current = i;
    var id = pages[i].id;

    navLinks.forEach(function (a) {
      a.setAttribute('aria-current', a.getAttribute('href') === '#' + id ? 'true' : 'false');
    });
    dots.forEach(function (d, j) {
      d.setAttribute('aria-current', j === i ? 'true' : 'false');
    });

    pages[i].classList.add('is-visible');

    if (id && history.replaceState) {
      history.replaceState(null, '', '#' + id);
    }
  }

  /* Active section = the one covering the middle of the viewport. */
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('is-visible');
        setActive(pages.indexOf(e.target));
      }
    });
  }, { root: scroller, threshold: 0.5 });

  pages.forEach(function (p) { io.observe(p); });

  /* ---- Scroll progress line ---- */
  function onScroll() {
    if (!bar) return;
    var max = scroller.scrollHeight - scroller.clientHeight;
    bar.style.width = (max > 0 ? (scroller.scrollTop / max) * 100 : 0) + '%';
  }
  scroller.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---- Quick-nav clicks (the scroller, not the window, does the scrolling) ---- */
  navLinks.forEach(function (a) {
    a.addEventListener('click', function (ev) {
      var target = document.querySelector(a.getAttribute('href'));
      if (!target) return;
      ev.preventDefault();
      goTo(pages.indexOf(target));
    });
  });

  /* ---- Keyboard paging ---- */
  document.addEventListener('keydown', function (e) {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;

    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;

    // Let a scrollable panel (the AI statement) consume arrows while it has room.
    var panel = t && t.closest ? t.closest('.panel') : null;
    if (panel && panel.scrollHeight > panel.clientHeight) {
      var atTop = panel.scrollTop <= 0;
      var atEnd = panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 1;
      if ((e.key === 'ArrowDown' && !atEnd) || (e.key === 'ArrowUp' && !atTop)) return;
    }

    switch (e.key) {
      case 'ArrowDown': case 'PageDown':
        e.preventDefault(); goTo(Math.min(current + 1, pages.length - 1)); break;
      case 'ArrowUp': case 'PageUp':
        e.preventDefault(); goTo(Math.max(current - 1, 0)); break;
      case 'Home':
        e.preventDefault(); goTo(0); break;
      case 'End':
        e.preventDefault(); goTo(pages.length - 1); break;
    }
  });

  /* ---- Honour a #hash on arrival (snap container ignores the native jump) ---- */
  if (location.hash) {
    var landing = document.querySelector(location.hash);
    if (landing && landing.classList.contains('page')) {
      requestAnimationFrame(function () {
        landing.scrollIntoView({ behavior: 'auto', block: 'start' });
      });
    }
  }

  setActive(0);
})();
