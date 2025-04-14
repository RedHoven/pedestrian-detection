document.addEventListener("DOMContentLoaded", () => {
  const toc = document.getElementById("toc");
  if (!toc) {
    console.error("TOC container not found");
    return;
  }

  // Get all h1, h2, and h3 headings (skip first two h1 elements)
  const headings = Array.from(document.querySelectorAll("h1, h2, h3"))
    .filter((heading, index) => {
      return !(heading.tagName === "H1" && index < 2);
    });

  // Generate TOC items with unique IDs
  const tocItems = headings.map((heading, index) => {
    const id = heading.id || `toc-${index}`;
    heading.id = id;
    return {
      id,
      text: heading.textContent.trim(),
      level: parseInt(heading.tagName.substring(1))
    };
  });

  // Group items by their parent h1
  const h1Groups = [];
  let currentGroup = null;

  tocItems.forEach(item => {
    if (item.level === 1) {
      currentGroup = { h1: item, children: [] };
      h1Groups.push(currentGroup);
    } else if (currentGroup) {
      currentGroup.children.push(item);
    }
  });

  // Build the TOC HTML structure
  function buildTOC(groups) {
    let html = "<ul class='toc-root'>";
    
    groups.forEach(group => {
      html += `
        <li class="h1-section">
          <a href="#${group.h1.id}" class="toc-link">${group.h1.text}</a>
          <ul class="h1-children">`;
      
      let levelStack = [1]; // Track nesting levels
      
      group.children.forEach(item => {
        while (levelStack.length && levelStack[levelStack.length - 1] >= item.level) {
          html += "</ul></li>";
          levelStack.pop();
        }
        
        html += `
          <li>
            <a href="#${item.id}" class="toc-link">${item.text}</a>
            <ul>`;
        levelStack.push(item.level);
      });
      
      // Close remaining tags
      while (levelStack.length > 1) {
        html += "</ul></li>";
        levelStack.pop();
      }
      
      html += `</ul></li>`;
    });
    
    return html + "</ul>";
  }

  toc.innerHTML = buildTOC(h1Groups);
  const tocLinks = toc.querySelectorAll(".toc-link");
  const h1Sections = toc.querySelectorAll(".h1-section");

  // Recursively expand all children of an element
  function expandAllChildren(element) {
    const uls = element.querySelectorAll("ul");
    uls.forEach(ul => {
      ul.style.display = "block";
      const parentLi = ul.closest("li");
      if (parentLi) parentLi.classList.add("expanded");
    });
  }

  // TOC activation logic
  function activateTOC() {
    const scrollPosition = window.scrollY + 100; // Activation offset
    let activeId = null;
    let closestDistance = Infinity;
    let activeIsH1 = false;

    // Find the heading closest to but above the scroll position
    tocItems.forEach(item => {
      const element = document.getElementById(item.id);
      if (element) {
        const distance = element.offsetTop - scrollPosition;
        if (distance <= 0 && Math.abs(distance) < Math.abs(closestDistance)) {
          closestDistance = distance;
          activeId = item.id;
          activeIsH1 = (item.level === 1);
        }
      }
    });

    // Reset all active states and collapse all h1 sections
    tocLinks.forEach(link => {
      link.classList.remove("active");
    });
    h1Sections.forEach(section => {
      section.classList.remove("expanded");
      const childUl = section.querySelector(".h1-children");
      if (childUl) childUl.style.display = "none";
    });

    // Set new active state
    if (activeId) {
      const activeLink = toc.querySelector(`a[href="#${activeId}"]`);
      if (activeLink) {
        activeLink.classList.add("active");

        // Handle the containing h1 section
        const containingH1 = activeLink.closest(".h1-section");
        if (containingH1) {
          containingH1.classList.add("expanded");
          const childUl = containingH1.querySelector(".h1-children");
          if (childUl) {
            childUl.style.display = "block";
            // If this is the active h1, expand all its children recursively
            if (activeIsH1) {
              expandAllChildren(containingH1);
            }
          }
        }

        // For non-h1 items, expand their entire parent chain
        if (!activeIsH1) {
          let parent = activeLink.parentElement;
          while (parent && !parent.classList.contains("toc-root")) {
            if (parent.tagName === "LI") {
              parent.classList.add("expanded");
              const childUl = parent.querySelector("> ul");
              if (childUl) {
                childUl.style.display = "block";
                // Expand all children of parent elements
                expandAllChildren(parent);
              }
            }
            parent = parent.parentElement;
          }
        }
      }
    }
  }

  // Initialize and set up events
  window.addEventListener("scroll", activateTOC);
  activateTOC(); // Initial activation

  // Toggle functionality for mobile
  const toggleBtn = document.getElementById("toc-toggle");
  const tocContainer = document.getElementById("toc-container");

  if (toggleBtn && tocContainer) {
    toggleBtn.addEventListener("click", () => {
      tocContainer.classList.toggle("visible");
    });

    document.addEventListener("click", (e) => {
      if (window.innerWidth <= 1250 &&
          !tocContainer.contains(e.target) &&
          !toggleBtn.contains(e.target)) {
        tocContainer.classList.remove("visible");
      }
    });
  }
});