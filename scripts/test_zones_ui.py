"""Offscreen check of the Zones page logic (no camera/models).

    python scripts\\test_zones_ui.py
"""
import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication, QMessageBox   # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
from bantayaso.ui import app as A                          # noqa: E402
from bantayaso.ui import theme as T                        # noqa: E402
from bantayaso.zones import Zone, ZoneEditor               # noqa: E402

fails = 0


def check(name, cond, info=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {info}" if info else ""))
    fails += 0 if cond else 1


T.apply("light")
app.setStyleSheet(T.QSS)
A.Worker.start = lambda self, *a, **k: None
w = A.MainWindow(argparse.Namespace(source=None, config="config.yaml", video=None, camera=None,
                                    no_hazards=True, no_actions=True, no_vlm=True))
copy = lambda zs: [Zone(z.name, z.type, [p[:] for p in z.points]) for z in zs]   # noqa: E731
saved = {"n": 0, "baked": 0, "zones": [Zone("Bed 1", "bed", [[.1, .5], [.5, .5], [.5, .9], [.1, .9]])]}
zones = copy(saved["zones"])
pipe = types.SimpleNamespace(zones=zones, editor=ZoneEditor(zones), aligner=types.SimpleNamespace(status="aligned"))
pipe.save_zones = lambda: saved.update(n=saved["n"] + 1, zones=copy(pipe.zones))
pipe.reload_zones = lambda: pipe.zones.__setitem__(slice(None), copy(saved["zones"]))
pipe.bake_alignment = lambda: saved.update(baked=saved["baked"] + 1)
w.worker.pipe = pipe
w.frame_size = (1000, 1000)
w.go(A.P_ZONES)
vis = lambda: sorted(k for k, b in w.zone_btn.items() if not b.isHidden())   # noqa: E731

check("view mode: only Edit zones, area bar hidden", vis() == ["edit"] and w.zone_toolbar.isHidden(), vis())
w.zone_click(300, 500, 1)
check("view mode: clicking the video does nothing", w.zone_sel is None and not pipe.editor.current)
w.zone_edit()
check("Edit zones: area bar + New/Clear/Cancel edits/Save appear", vis() == ["clear", "discard", "new", "save"]
      and not w.zone_toolbar.isHidden() and pipe.editor.editing, vis())
check("Edit snaps zones to the camera's current view", saved["baked"] == 1)
w.zone_click(300, 700, 1)
check("clicking INSIDE a zone does nothing (only corners)", w.zone_sel is None)
w.zone_hover(101, 502)
check("hovering a corner highlights it", pipe.editor.hover == (0, 0))
w.zone_hover(300, 700)
check("hover away clears it", pipe.editor.hover is None)
w.zone_click(500, 900, 1); w.zone_drag(600, 950); w._zone_drag = None
check("grab a corner: selects the zone and drags that corner", w.zone_sel == 0 and zones[0].points[2] == [0.6, 0.95])
check("selected: Delete zone + Cancel edits + Save + New", vis() == ["delete", "discard", "new", "save"], vis())
w.zone_name.setText("Big bed"); w.zone_rename("Big bed")
check("name box renames the selected zone", zones[0].name == "Big bed")
w.zone_click(100, 900, 2)
check("right-click a corner removes it", len(zones[0].points) == 3)
w.zone_click(100, 500, 2)
check("a zone keeps at least 3 corners", len(zones[0].points) == 3)
w.zone_new()
check("New zone: Undo/Cancel/Finish only", vis() == ["cancel", "finish", "undo"], vis())
check("Finish disabled with no corners", not w.zone_btn["finish"].isEnabled())
w.zone_type.button(A.ZONE_TYPES.index("danger")).click()
check("area chip sets the new zone's type (old zone unchanged)", pipe.editor.type == "danger" and zones[0].type == "bed")
for x, y in ((700, 100), (900, 100), (900, 300), (700, 300)):
    w.zone_click(x, y, 1)
w.zone_undo()
check("Undo point", len(pipe.editor.current) == 3)
w.zone_click(702, 101, 1)
check("click the first corner closes it; new zone selected", len(zones) == 2 and zones[1].type == "danger"
      and w.zone_sel == 1, [z.name for z in zones])
chips = [w.zone_chips.itemAt(i).widget().text() for i in range(w.zone_chips.count())]
check("numbered zone buttons", chips == ["1  Big bed", "2  Danger 1"], chips)
w._chip_clicked(0)
check("click a numbered button selects that zone", w.zone_sel == 0)
w.zone_type.button(A.ZONE_TYPES.index("food")).click()
check("area chip changes the selected zone's type", zones[0].type == "food")
w.zone_delete()
check("Delete zone removes it", [z.name for z in zones] == ["Danger 1"] and w.zone_dirty)
w.zone_save()
check("Save: writes, back to view mode", saved["n"] == 1 and w.zone_mode == "view" and vis() == ["edit"]
      and [z.name for z in saved["zones"]] == ["Danger 1"])
w._chip_clicked(0)
check("numbered button in view mode opens edit with it selected", w.zone_mode == "edit" and w.zone_sel == 0)
w.zone_click(700, 100, 1); w.zone_drag(650, 80); w._zone_drag = None
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
w.zone_discard_exit()
check("Cancel edits restores the saved zones", zones[0].points[0] == [0.7, 0.1] and w.zone_mode == "view",
      zones[0].points[0])
w.zone_edit(); w.zone_click(700, 100, 1); w.zone_drag(650, 80); w._zone_drag = None
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Cancel)
w.go(A.P_MONITOR)
check("leaving with unsaved edits -> Cancel stays", w.stack.currentIndex() == A.P_ZONES)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Save)
w.go(A.P_MONITOR)
check("leaving -> Save saves and leaves", saved["n"] == 2 and w.stack.currentIndex() == A.P_MONITOR
      and w.zone_mode == "view")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
