// Tiny vanilla carousel — no deps.
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-carousel]').forEach(carousel => {
    const slides = carousel.querySelectorAll('.carousel-slide');
    const dots = carousel.querySelectorAll('.carousel-dot');
    if (!slides.length) return;
    let current = 0;
    let timer = null;

    function show(i) {
      slides[current].classList.remove('is-active');
      if (dots[current]) dots[current].classList.remove('is-active');
      current = (i + slides.length) % slides.length;
      slides[current].classList.add('is-active');
      if (dots[current]) dots[current].classList.add('is-active');
    }

    function autoplay() {
      stop();
      timer = setInterval(() => show(current + 1), 7000);
    }
    function stop() {
      if (timer) clearInterval(timer);
      timer = null;
    }

    const prev = carousel.querySelector('.carousel-prev');
    const next = carousel.querySelector('.carousel-next');
    if (prev) prev.addEventListener('click', () => { show(current - 1); autoplay(); });
    if (next) next.addEventListener('click', () => { show(current + 1); autoplay(); });
    dots.forEach((d, i) => d.addEventListener('click', () => { show(i); autoplay(); }));

    carousel.addEventListener('mouseenter', stop);
    carousel.addEventListener('mouseleave', autoplay);

    // Keyboard navigation when carousel is focused
    carousel.tabIndex = 0;
    carousel.addEventListener('keydown', e => {
      if (e.key === 'ArrowLeft')  { show(current - 1); autoplay(); }
      if (e.key === 'ArrowRight') { show(current + 1); autoplay(); }
    });

    autoplay();
  });
});
