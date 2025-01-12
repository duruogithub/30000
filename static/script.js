document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const inputs = document.querySelectorAll("select");

    form.addEventListener("submit", function (e) {
        let valid = true;
        inputs.forEach(input => {
            const errorText = input.nextElementSibling; // 错误提示位置
            if (!input.value) {
                input.style.border = "1px solid red";
                if (!errorText) {
                    input.insertAdjacentHTML(
                        "afterend",
                        `<small>This field is required.</small>`
                    );
                }
                valid = false;
            } else {
                input.style.border = "1px solid #ddd";
                if (errorText) {
                    errorText.remove();
                }
            }
        });
        if (!valid) {
            e.preventDefault();
        }
    });

    const translations = {
        en: {
            gender: "Gender",
            age: "Age",
            bmi: "BMI",
            residence: "Residence",
            fx: "History of chronic diarrhea",
            bm: "History of chronic constipation",
            lwy: "History of chronic appendicitis or appendectomy",
            smoke: "Smoking status",
            drink: "Alcohol consumption",
            fit: "FIT Test Result",
            submit: "Submit",
        },
        cn: {
            gender: "性别",
            age: "年龄",
            bmi: "体重指数 (BMI)",
            residence: "居住地",
            fx: "慢性腹泻史",
            bm: "慢性便秘史",
            lwy: "慢性阑尾炎或阑尾切除史",
            smoke: "吸烟状态",
            drink: "饮酒情况",
            fit: "粪便隐血试验结果",
            submit: "提交",
        },
    };

    const savedLang = localStorage.getItem("lang") || "en";
    updateLanguage(savedLang);

    document.querySelectorAll("#lang-en, #lang-cn").forEach(button => {
        button.addEventListener("click", function () {
            const lang = this.id === "lang-en" ? "en" : "cn";
            localStorage.setItem("lang", lang);
            updateLanguage(lang);
        });
    });

    function updateLanguage(lang) {
        document.querySelectorAll("label").forEach(label => {
            const forAttr = label.getAttribute("for");
            if (translations[lang][forAttr]) {
                label.innerText = translations[lang][forAttr] + ":";
            }
        });
        document.querySelector("button[type='submit']").innerText = translations[lang].submit;
    }
});