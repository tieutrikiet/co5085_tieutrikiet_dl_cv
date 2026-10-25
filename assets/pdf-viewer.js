/* CO5085 — popup PDF viewer for [data-pdf-open] buttons. */
(function () {
  'use strict';

  var modal = document.querySelector('.pdf-modal');
  if (!modal || typeof modal.showModal !== 'function') return;

  var frame    = modal.querySelector('.pdf-modal-frame');
  var title    = modal.querySelector('.pdf-modal-title');
  var download = modal.querySelector('.pdf-modal-download');

  document.addEventListener('click', function (e) {
    var btn = e.target.closest('[data-pdf-open]');
    if (!btn) return;
    var src = btn.getAttribute('data-pdf-open');
    title.textContent = btn.getAttribute('data-pdf-title') || '';
    download.href = src;
    frame.src = src;
    modal.showModal();
  });

  modal.querySelector('.pdf-modal-close').addEventListener('click', function () { modal.close(); });

  /* Close when clicking the backdrop (outside the dialog box). */
  modal.addEventListener('click', function (e) {
    if (e.target === modal) modal.close();
  });

  /* Drop the PDF on close so it does not keep rendering in the background. */
  modal.addEventListener('close', function () { frame.removeAttribute('src'); });
})();
