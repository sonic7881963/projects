
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level6/right1.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '06a79I1MFlKZYzGvjTIzIZ3', 'right1');
// 3.16小游戏/command_TypeScript/level6/right1.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___002_1 = require("./global - 002");
var enemy_1 = /** @class */ (function (_super) {
    __extends(enemy_1, _super);
    function enemy_1() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_1.prototype.start = function () {
        this.schedule(function () {
            global___002_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_1.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___002_1.default.minion_attack == true) {
            global___002_1.default.main_hp -= 20;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_1.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global___002_1.default.minion_attack = false;
        if (global___002_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___002_1.default.main_hp, 19);
        }
    };
    enemy_1.prototype.update = function (dt) {
        if (global___002_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy_1 = __decorate([
        ccclass
    ], enemy_1);
    return enemy_1;
}(cc.Component));
exports.default = enemy_1;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDZcXHJpZ2h0MS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7O0FBQ0Esb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUU1RSxJQUFBLEtBQXNCLEVBQUUsQ0FBQyxVQUFVLEVBQWxDLE9BQU8sYUFBQSxFQUFFLFFBQVEsY0FBaUIsQ0FBQztBQUMxQywrQ0FBbUM7QUFLbkM7SUFBcUMsMkJBQVk7SUFBakQ7O0lBNERBLENBQUM7SUF2REcsd0JBQU0sR0FBTjtRQUNJLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsbUJBQW1CLEVBQUUsQ0FBQztRQUNoRCxPQUFPLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQztJQUMzQixDQUFDO0lBR0QsdUJBQUssR0FBTDtRQUNJLElBQUksQ0FBQyxRQUFRLENBQUM7WUFDVixzQkFBTSxDQUFDLGFBQWEsR0FBRyxJQUFJLENBQUM7UUFDaEMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO1FBRUwsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQUVELGtDQUFnQixHQUFoQixVQUFpQixLQUFLLEVBQUMsSUFBSTtRQUN2QixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sR0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDekIsSUFBRyxzQkFBTSxDQUFDLGFBQWEsSUFBSSxJQUFJLEVBQUU7WUFDN0Isc0JBQU0sQ0FBQyxPQUFPLElBQUksRUFBRSxDQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNwRCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN2RCxPQUFPLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDO1NBQzlDO0lBRUwsQ0FBQztJQUVELGlDQUFlLEdBQWYsVUFBZ0IsS0FBSztRQUNsQixFQUFFLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBR2Ysc0JBQU0sQ0FBQyxhQUFhLEdBQUcsS0FBSyxDQUFDO1FBQzdCLElBQUksc0JBQU0sQ0FBQyxPQUFPLElBQUksQ0FBQyxFQUFFO1lBQ3hCLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLEtBQUssQ0FBQztZQUMxQixJQUFJLElBQUksR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUM7WUFDckMsSUFBSSxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUM7U0FDbkI7YUFBTTtZQUNOLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBRSxzQkFBTSxDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsQ0FBQztTQUMzRDtJQUVKLENBQUM7SUFHRCx3QkFBTSxHQUFOLFVBQVEsRUFBRTtRQUVOLElBQUksc0JBQU0sQ0FBQyxhQUFhLElBQUssSUFBSSxFQUFFO1lBQy9CLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBSSxJQUFJLEdBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBRSxDQUFDO1NBR2hGO2FBQU07WUFDSCxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUssSUFBSSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSyxJQUFJLENBQUMsUUFBUSxHQUFHLEVBQUUsQ0FBQyxFQUFJO2dCQUN2RyxJQUFJLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMzRTtTQUVKO0lBQ1IsQ0FBQztJQTNEZ0IsT0FBTztRQUQzQixPQUFPO09BQ2EsT0FBTyxDQTREM0I7SUFBRCxjQUFDO0NBNURELEFBNERDLENBNURvQyxFQUFFLENBQUMsU0FBUyxHQTREaEQ7a0JBNURvQixPQUFPIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiXHJcbi8vIExlYXJuIFR5cGVTY3JpcHQ6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3R5cGVzY3JpcHQuaHRtbFxyXG4vLyBMZWFybiBBdHRyaWJ1dGU6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL3JlZmVyZW5jZS9hdHRyaWJ1dGVzLmh0bWxcclxuLy8gTGVhcm4gbGlmZS1jeWNsZSBjYWxsYmFja3M6XHJcbi8vICAtIGh0dHBzOi8vZG9jcy5jb2Nvcy5jb20vY3JlYXRvci9tYW51YWwvZW4vc2NyaXB0aW5nL2xpZmUtY3ljbGUtY2FsbGJhY2tzLmh0bWxcclxuXHJcbmNvbnN0IHtjY2NsYXNzLCBwcm9wZXJ0eX0gPSBjYy5fZGVjb3JhdG9yO1xyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbCAtIDAwMlwiXHJcblxyXG5cclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGVuZW15XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG4gICBcclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG4gICAgbWluaW9uX3g7XHJcbiAgICBcclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIHN0YXJ0ICgpIHtcclxuICAgICAgICB0aGlzLnNjaGVkdWxlKCgpID0+IHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSB0cnVlO1xyXG4gICAgICAgIH0sMSk7XHJcblxyXG4gICAgICAgIHRoaXMubWluaW9uX3ggPSB0aGlzLm5vZGUucG9zaXRpb24ueDtcclxuICAgIH1cclxuICAgIFxyXG4gICAgb25Db2xsaXNpb25FbnRlcihvdGhlcixzZWxmKXtcclxuICAgICAgICBjYy5sb2coXCLlvIDlp4vnorDmkp5cIitvdGhlci50YWcpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5taW5pb25fYXR0YWNrID09IHRydWUpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1haW5faHAgLT0gMjA7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4vZW5lbXlfZGFtYWdlXCIpO1xyXG4gICAgICAgICAgICBkYW1hZ2UuZ2V0Q29tcG9uZW50KGNjLkxhYmVsKS5zdHJpbmcgPSBcIi0yMFwiO1xyXG4gICAgICAgICAgICBsZXQgZGFtYWdlMiA9IGNjLmZpbmQoXCJDYW52YXMvYmova3Vhbi94eS9tYWluX2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgfVxyXG4gICAgICAgXHJcbiAgICB9XHJcblxyXG4gICAgb25Db2xsaXNpb25FeGl0KG90aGVyKSB7XHJcbiAgICAgICBjYy5sb2coXCLnorDmkp7nu5PmnZ9cIik7XHJcbiAgICAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgIGlmKCBnbG9iYWwubWFpbl9ocCA8PSAwKSB7XHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgICBsZXQgbG9zZSA9IGNjLmZpbmQoXCJDYW52YXMvYmovZmFpbFwiKTtcclxuICAgICAgICBsb3NlLmFjdGl2ZSA9IHRydWU7XHJcbiAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoIGdsb2JhbC5tYWluX2hwLCAxOSk7XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuXHJcblxyXG4gICAgdXBkYXRlIChkdCkge1xyXG5cclxuICAgICAgICBpZiAoZ2xvYmFsLm1pbmlvbl9hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKCAgdGhpcy5ub2RlLnBvc2l0aW9uLnggIC0gMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkgKTtcclxuICAgICAgICAgICAgICAgXHJcbiAgICAgICAgICAgXHJcbiAgICAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICAgICAgaWYgKCEodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gIHRoaXMubWluaW9uX3ggKyA1MCkgJiYgKHRoaXMubm9kZS5wb3NpdGlvbi54IDw9ICB0aGlzLm1pbmlvbl94IC0gNTApKSAgIHtcclxuICAgICAgICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgIFxyXG4gICAgICAgICAgIH1cclxuICAgIH1cclxufVxyXG4iXX0=