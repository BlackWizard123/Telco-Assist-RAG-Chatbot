document.addEventListener('DOMContentLoaded', function() {

	var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bss-tooltip]'));
	var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
	  return new bootstrap.Tooltip(tooltipTriggerEl);
	})

	var toastTriggers = document.querySelectorAll('[data-bs-toggle="toast"]');

	for (var toastTrigger of toastTriggers) {
		toastTrigger.addEventListener('click', function () {
			var toastSelector = toastTrigger.getAttribute('data-bs-target');

			if (!toastSelector) return;

			try {
				var toastEl = document.querySelector(toastSelector);

				if (!toastEl) return;

				var toast = new bootstrap.Toast(toastEl);
				toast.show();
			}
			catch(e) {
				console.error(e);
			}
		})
	}
}, false);