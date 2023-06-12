async function inpaintAnything_sendToInpaint() {

	const waitForElement = async (
	    parent,
	    selector,
	    exist
	) => {
	    return new Promise((resolve) => {
	        const observer = new MutationObserver(() => {
	            if (!!parent.querySelector(selector) != exist) {
	                return;
	            }
	            observer.disconnect();
	            resolve(undefined);
	        })

	        observer.observe(parent, {
	            childList: true,
	            subtree: true,
	        })

	        if (!!parent.querySelector(selector) == exist) {
	            resolve(undefined);
	        }
	    })
	}

	const timeout = (ms) => {
	    return new Promise(function (resolve, reject) {
	        setTimeout(() => reject('Timeout'), ms)
	    })
	};

	const waitForElementToBeInDocument = (parent, selector) => Promise.race([waitForElement(parent, selector, true), timeout(10000)]);

	const waitForElementToBeRemoved = (parent, selector) => Promise.race([waitForElement(parent, selector, false), timeout(10000)]);

	const updateGradioImage = async (element, url, name) => {
	    const blob = await (await fetch(url)).blob();
	    const file = new File([blob], name);
	    const dt = new DataTransfer();
	    dt.items.add(file);

	    element
	        .querySelector("button[aria-label='Clear']")
	        ?.click();
	    await waitForElementToBeRemoved(element, "button[aria-label='Clear']");
	    const input = element.querySelector("input[type='file']");
	    input.value = '';
	    input.files = dt.files;
	    input.dispatchEvent(
	        new Event('change', {
	            bubbles: true,
	            composed: true,
	        })
	    );
	    await waitForElementToBeInDocument(element, "button[aria-label='Clear']");
	}

	const inputImg = document.querySelector("#input_image img");
	const maskImg = document.querySelector("#mask_out_image img");

	if (!inputImg || !maskImg) {
		return;
	}

	const inputImgDataUrl = inputImg.src;
	const maskImgDataUrl = maskImg.src;

	window.scrollTo(0, 0);
	switch_to_img2img_tab(4);

	await waitForElementToBeInDocument(document.querySelector("#img2img_inpaint_upload_tab"), "#img_inpaint_base");

	await updateGradioImage(document.querySelector("#img_inpaint_base"), inputImgDataUrl, "input.png");
	await updateGradioImage(document.querySelector("#img_inpaint_mask"), maskImgDataUrl, "mask.png");
}

async function inpaintAnything_clearSamMask() {
	await new Promise(s => setTimeout(s, 300));

	const sam_mask_clear = document.querySelector("#sam_image").querySelector("button[aria-label='Clear']");
	if (!sam_mask_clear) {
		return;
	}
	sam_mask_clear.dispatchEvent(
		new Event('click', {
			bubbles: true,
			composed: true,
		})
	);
}

async function inpaintAnything_clearSelMask() {
	await new Promise(s => setTimeout(s, 300));

	const sel_mask_clear = document.querySelector("#sel_mask").querySelector("button[aria-label='Clear']");
	if (!sel_mask_clear) {
		return;
	}
	sel_mask_clear.dispatchEvent(
		new Event('click', {
			bubbles: true,
			composed: true,
		})
	);
}