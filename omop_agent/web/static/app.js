// Small progressive enhancements; every page works without this file.
document.addEventListener("submit", function (event) {
  var form = event.target;
  var button = event.submitter;
  var message = (button && button.getAttribute("data-confirm")) || form.getAttribute("data-confirm");
  if (message && !window.confirm(message)) {
    event.preventDefault();
  }
});
