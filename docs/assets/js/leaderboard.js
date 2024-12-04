function sortTable(columnIndex) {
    var table = document.getElementById("sortableTable");
    var rows = Array.from(table.getElementsByTagName("tr")).slice(1);  // Skip header row
    var isAscending = table.rows[0].cells[columnIndex].getAttribute("data-sort-order") === "asc";
    rows.sort(function(rowA, rowB) {
        var cellA = rowA.cells[columnIndex].textContent.trim();
        var cellB = rowB.cells[columnIndex].textContent.trim();
        if (!isNaN(cellA) && !isNaN(cellB)) {
        return isAscending ? cellA - cellB : cellB - cellA;
        } else {
        return isAscending ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
        }
    });
    rows.forEach(function(row) {
        table.appendChild(row);
    });
    table.rows[0].cells[columnIndex].setAttribute("data-sort-order", isAscending ? "desc" : "asc");
}