"""Offscreen check of the Zones page logic (no camera/models): draw, finish, select, reshape,
move, rename, retype, remove a corner, delete, clear, save, discard, leave-page guard.

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
A.Worker.start = lambda self, *a, **k: None                # no camera / models
w = A.MainWindow(argparse.Namespace(source=None, config="config.yaml", video=None, camera=None,
                                    no_hazards=True, no_actions=True, no_vlm=True))
saved = {"n": 0, "zones": [Zone("Bed 1", "bed", [[.1, .5], [.5, .5], [.5, .9], [.1, .9]])]}
zones = [Zone(z.name, z.type, [p[:] for p in z.points]) for z in saved["zones"]]
pipe = types.SimpleNamespace(zones=zones, editor=ZoneEditor(zones))
pipe.save_zones = lambda: saved.update(n=saved["n"] + 1, zones=[Zone(z.name, z.type, [p[:] for p in z.points]) for z in pipe.zones])
pipe.reload_zones = lambda: pipe.zones.__setitem__(slice(None), [Zone(z.name, z.type, [p[:] for p in z.points]) for z in saved["zones"]])
w.worker.pipe = pipe
w.frame_size = (1000, 1000)
w.go(A.P_ZONES)
vis = lambda: sorted(k for k, b in w.zone_btn.items() if not b.isHidden())   # noqa: E731

check("idle with zones: only Clear all + Save", vis() == ["clear", "save"], vis())
check("Save disabled when nothing changed", not w.zone_btn["save"].isEnabled())
# click empty space -> start drawing
w.zone_click(700, 100, 1)
check("click empty spot starts a new zone", len(pipe.editor.current) == 1 and vis() == ["cancel", "finish", "save", "undo"], vis())
check("Finish disabled under 3 corners", not w.zone_btn["finish"].isEnabled())
w.zone_click(900, 100, 1); w.zone_click(900, 300, 1)
check("Finish enabled at 3 corners", w.zone_btn["finish"].isEnabled())
w.zone_undo()
check("Undo point removes the last corner", len(pipe.editor.current) == 2)
w.zone_click(900, 300, 1); w.zone_click(700, 300, 1)
w.zone_click(702, 101, 1)                                   # click the first corner again
check("clicking the first corner closes the zone", len(zones) == 2 and not pipe.editor.current, [z.name for z in zones])
check("new zone is selected for adjusting", w.zone_sel == 1 and vis() == ["delete", "done", "save"], vis())
check("unsaved changes shown, Save enabled", w.zone_dirty and w.zone_btn["save"].isEnabled()
      and "Unsaved" in w.zone_dirty_lbl.text())
# reshape: drag a corner
w.zone_click(900, 100, 1); w.zone_drag(950, 50); w._zone_drag = None
check("drag a dot moves that corner", zones[1].points[1] == [0.95, 0.05], zones[1].points[1])
# move whole zone
before = [p[:] for p in zones[1].points]
w.zone_click(800, 200, 1); w.zone_drag(800, 250); w._zone_drag = None
check("drag inside moves the whole zone", all(abs(b[1] + .05 - a[1]) < 1e-6 for a, b in zip(zones[1].points, before)))
# rename + retype
w.zone_name.setText("Sofa"); w.zone_rename("Sofa")
check("typing a name renames the selected zone", zones[1].name == "Sofa")
w.zone_type.button(A.ZONE_TYPES.index("danger")).click()
check("area chip changes the selected zone's type", zones[1].type == "danger")
# remove a corner (right-click) down to 3 then refuse
w.zone_click(700, 350, 2)
check("right-click a dot removes that corner", len(zones[1].points) == 3, len(zones[1].points))
w.zone_click(int(zones[1].points[0][0] * 1000), int(zones[1].points[0][1] * 1000), 2)
check("cannot go below 3 corners", len(zones[1].points) == 3)
# done / select another by clicking it / chips
w.zone_deselect()
check("Done stops editing", w.zone_sel is None and vis() == ["clear", "save"], vis())
w.zone_click(300, 700, 1)
check("click inside a zone selects it", w.zone_sel == 0)
w.zone_click(50, 50, 1)
check("click outside while editing just deselects (no new zone)", w.zone_sel is None and not pipe.editor.current)
# save
w.zone_save()
check("Save writes zones and clears the unsaved mark", saved["n"] == 1 and not w.zone_dirty
      and [z.name for z in saved["zones"]] == ["Bed 1", "Sofa"])
# delete then discard by leaving the page
w._select_zone(1); w.zone_delete()
check("Delete zone removes only the selected zone", [z.name for z in zones] == ["Bed 1"] and w.zone_dirty)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
w.go(A.P_MONITOR)
check("leaving with unsaved changes -> Discard restores the saved zones",
      [z.name for z in zones] == ["Bed 1", "Sofa"] and w.stack.currentIndex() == A.P_MONITOR)
w.go(A.P_ZONES)
w._select_zone(0); w.zone_delete()
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Cancel)
w.go(A.P_MONITOR)
check("Cancel keeps you on the Zones page", w.stack.currentIndex() == A.P_ZONES)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
w.zone_clear()
check("Clear all empties the list (until saved)", zones == [] and w.zone_dirty)
w.zone_click(100, 100, 1); w.zone_cancel()
check("Cancel throws away a half-drawn zone", not pipe.editor.current)

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
